import re
from typing import Iterable, Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers import TokenizerLike


@ReasoningParserManager.register_module(name="olala")
class OlalaParser(ReasoningParser):
    """Reasoning parser for the Olala channel-based chat format."""

    # These appear literally in the model output in both the enabled and disabled
    # cases, making them reliable markers for reasoning-end detection.
    _FINAL_MARKER = "<|channel_start|>final<|content|>"
    _TOOLS_MARKER = "<|channel_start|>tools<|content|>"
    _CHANNEL_END = "<|channel_end|>"

    # Mirrors OlalaToolParser.COMPLETE_CALL_RE. Only a tools channel holding at
    # least one complete <call> is worth forwarding — see extract_reasoning.
    _COMPLETE_CALL_RE = re.compile(
        r'<call\s+id="[^"]*">\s*<name>.*?</name>'
        r"\s*<arguments>.*?</arguments>\s*</call>",
        re.DOTALL,
    )

    @property
    def reasoning_start_str(self) -> str:
        return "<|channel_start|>analysis<|content|>"

    @property
    def reasoning_end_str(self) -> str:
        # <|channel_start|>final<|content|> always appears in generated output:
        # - enabled: right after <|channel_end|> that closes analysis
        # - disabled: as the very first generated token sequence
        return self._FINAL_MARKER

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self._channel_start_id = self.vocab["<|channel_start|>"]
        self._channel_end_id = self.vocab["<|channel_end|>"]
        self._content_id = self.vocab["<|content|>"]
        self._eom_token_id = self.vocab["<|im_end|>"]

        # encode() with add_special_tokens=False avoids a leading BOS disrupting
        # mid-stream sequence matching.
        self._reasoning_end_ids = self.model_tokenizer.encode(
            self.reasoning_end_str, add_special_tokens=False
        )
        self._content_start_ids = self._reasoning_end_ids  # same sequence

        # Streaming state
        self._state = "write-channel"  # write-channel | open-channel | idle
        self._current_channel = "analysis"
        self._buffer: list[int] = []

    # ------------------------------------------------------------------
    # xgrammar phase-gating
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Return True once <|channel_start|>final<|content|> has been generated.

        Works for both the enabled case (explicit end token precedes it) and the
        disabled case (it is the very first generated sequence).
        """
        n = len(self._reasoning_end_ids)
        for i in range(len(input_ids) - n, -1, -1):
            if input_ids[i] == self._eom_token_id:
                return False
            if list(input_ids[i : i + n]) == self._reasoning_end_ids:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Return token ids from the start of the final channel to end of output."""
        n = len(self._content_start_ids)
        for i in range(len(input_ids) - n, -1, -1):
            if input_ids[i] == self._eom_token_id:
                return []
            if list(input_ids[i : i + n]) == self._content_start_ids:
                return list(input_ids[i + n :])
        return []

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count tokens before the first channel boundary.

        For the enabled case this counts analysis tokens (before <|channel_end|>).
        For the disabled case the first token is <|channel_start|> → returns 0.
        """
        count = 0
        for tok_id in token_ids:
            if tok_id in (
                self._channel_start_id,
                self._channel_end_id,
                self._eom_token_id,
            ):
                return count
            count += 1
        return count

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _tools_expected(request) -> bool:
        """Whether the tool parser will run on this request's content.

        Only then is it safe to keep the tools channel in `content`; otherwise
        nothing downstream would strip the markers and they would leak to the
        user.
        """
        if request is None:
            return True
        if not getattr(request, "tools", None):
            return False
        return getattr(request, "tool_choice", None) != "none"

    def extract_reasoning(
        self,
        model_output: str,
        request,
    ) -> tuple[str | None, str | None]:
        if self._FINAL_MARKER not in model_output:
            # No final channel yet — all output is analysis (or model timed out).
            return model_output.strip() or None, None

        analysis_part, after_final = model_output.split(self._FINAL_MARKER, 1)

        # Analysis ends at its <|channel_end|>; trim that and any trailing space.
        analysis = analysis_part.split(self._CHANNEL_END)[0].strip() or None

        # Final content ends at its <|channel_end|> (before the tools channel or EOM).
        final = after_final.split(self._CHANNEL_END)[0].strip() or None

        # The tools channel sits AFTER the <|channel_end|> that closes the final
        # channel, so the split above drops it. That matters because
        # DelegatingParser.parse() feeds the tool parser this `content` string,
        # NOT the raw model output:
        #     reasoning, content = self.extract_reasoning(model_output, request)
        #     tool_calls, content = self._extract_tool_calls(content=content, ...)
        # (vllm/parser/abstract_parser.py). Dropping the tools channel here
        # therefore made every non-streaming tool call vanish: the tool parser
        # never saw <|channel_start|>tools<|content|> and reported
        # tools_called=False. Re-attach it after the final text; the tool parser
        # splits on the marker and strips it back off.
        #
        # Forward it only when it holds a complete <call>. When no tool call can
        # be extracted, _extract_tool_calls() returns the content it was *given*
        # and throws away the cleaned content the tool parser computed
        # ("return None, content" in the no-tool-calls branch) — so a tools
        # channel that is empty or truncated mid-call would leak its raw markers
        # straight to the user.
        if self._TOOLS_MARKER in after_final and self._tools_expected(request):
            tools_channel = (
                self._TOOLS_MARKER + after_final.split(self._TOOLS_MARKER, 1)[1]
            )
            if self._COMPLETE_CALL_RE.search(tools_channel):
                final = f"{final}{tools_channel}" if final else tools_channel

        return analysis, final

    # ------------------------------------------------------------------
    # Streaming extraction
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        buffers: dict[str, list[int]] = {"analysis": [], "final": []}

        for tok_id in delta_token_ids:
            if tok_id == self._channel_start_id:
                self._state = "open-channel"
                self._buffer.clear()
            elif tok_id == self._channel_end_id:
                # Explicit close of the current channel.
                self._state = "idle"
            elif tok_id == self._content_id:
                self._current_channel = self.model_tokenizer.decode(self._buffer)
                self._state = "write-channel"
                self._buffer.clear()
            elif self._state == "open-channel":
                self._buffer.append(tok_id)
            elif self._state == "write-channel":
                if (b := buffers.get(self._current_channel)) is not None:
                    b.append(tok_id)

        reasoning_delta = (
            self.model_tokenizer.decode(buffers["analysis"])
            if buffers["analysis"]
            else None
        )
        content_delta = (
            self.model_tokenizer.decode(buffers["final"]) if buffers["final"] else None
        )

        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    # ------------------------------------------------------------------
    # Request adjustment
    # ------------------------------------------------------------------

    def adjust_request(self, request):
        request.skip_special_tokens = False
        return request

    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: ToolServer | None,
    ) -> str | None:
        return None
