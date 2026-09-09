import re
from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser


@ToolParserManager.register_module(name="olala")
class OlalaToolParser(ToolParser):
    """Tool-call parser for the Olala channel-based chat format.

    Assistant turn structure (model output after the pre-filled prefix):
        {analysis}<|channel_end|>
        <|channel_start|>final<|content|>{text}<|channel_end|>
        [<|channel_start|>tools<|content|><toolcalls>
            <call id="..."><name>fn</name><arguments>{"k":"v"}</arguments></call>
        </toolcalls><|channel_end|>]
        <|im_end|>
    """

    TOOLS_MARKER = "<|channel_start|>tools<|content|>"
    FINAL_MARKER = "<|channel_start|>final<|content|>"
    CHANNEL_END = "<|channel_end|>"

    ARGS_OPEN = "<arguments>"
    ARGS_CLOSE = "</arguments>"

    # This format is XML, not JSON, so vLLM's generic "required"/named
    # tool_choice handling cannot work on it: it either feeds `content` to
    # TypeAdapter(list[FunctionDefinition]).validate_json() ("required") or
    # copies `content` verbatim into `arguments` (named). Both produce garbage
    # here. Declaring False makes vLLM route those cases through the auto
    # parsing path below instead. NB there is no grammar backing this format, so
    # "required"/named are best-effort, not guaranteed.
    supports_required_and_named = False

    # Matches a fully closed <call>...</call> block.
    COMPLETE_CALL_RE = re.compile(
        r'<call\s+id="([^"]*)">'
        r"\s*<name>(.*?)</name>"
        r"\s*<arguments>(.*?)</arguments>"
        r"\s*</call>",
        re.DOTALL,
    )

    CALL_OPEN_RE = re.compile(r'<call\s+id="([^"]*)">')
    NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)
        # Per-stream state: chars of `arguments` already sent per call (by index).
        self._sent_args_chars: list[int] = []
        # Ids synthesised for calls the model emitted without one, kept stable
        # across deltas (make_tool_call_id() would return a new id per chunk).
        self._synth_ids: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tools_content(self, text: str) -> str | None:
        """Return raw tools-channel text, or None if the channel hasn't opened yet.

        The tools channel is explicitly closed by <|channel_end|>, so we stop
        there rather than scanning for the next opener or EOM.
        """
        if self.TOOLS_MARKER not in text:
            return None
        raw = text.split(self.TOOLS_MARKER, 1)[1]
        end = raw.find(self.CHANNEL_END)
        if end != -1:
            raw = raw[:end]
        return raw

    def _final_content(self, text: str) -> str | None:
        """Return user-facing text from the final channel, or None.

        Handles both strings this can be called with:
          * the raw model output, still carrying FINAL_MARKER; and
          * the `content` the reasoning parser hands down, whose final-channel
            wrapper is already stripped and which is just
            "<final text><tools channel>".
        """
        if self.FINAL_MARKER in text:
            raw = text.split(self.FINAL_MARKER, 1)[1]
            end = raw.find(self.CHANNEL_END)
            if end != -1:
                raw = raw[:end]
        else:
            raw = text.split(self.TOOLS_MARKER, 1)[0]
            if raw.endswith(self.CHANNEL_END):
                raw = raw[: -len(self.CHANNEL_END)]
        return raw.strip() or None

    @staticmethod
    def _held_back(text: str, tag: str) -> int:
        """Length of the trailing run of `text` that is a proper prefix of `tag`.

        Used to withhold a half-generated "</arguments>" so partial closing
        markup never leaks into a streamed argument delta.
        """
        for n in range(min(len(text), len(tag) - 1), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0

    def _args_so_far(self, after_name: str) -> str | None:
        """`arguments` text emitted so far for one call, or None if not open yet.

        Cuts at </arguments> once it appears and holds back any partial closing
        tag, so the value only ever grows and is always a prefix of the final
        argument string. The previous implementation used a greedy
        `<arguments>(.*)` that swallowed "</arguments></call>" as argument text,
        which is what made streamed `arguments` invalid JSON.
        """
        idx = after_name.find(self.ARGS_OPEN)
        if idx == -1:
            return None
        args = after_name[idx + len(self.ARGS_OPEN) :]
        close = args.find(self.ARGS_CLOSE)
        if close != -1:
            return args[:close]
        held = self._held_back(args, self.ARGS_CLOSE)
        return args[: len(args) - held] if held else args

    def _call_id(self, index: int, raw_id: str) -> str:
        if raw_id:
            return raw_id
        if index not in self._synth_ids:
            self._synth_ids[index] = make_tool_call_id()
        return self._synth_ids[index]

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> ExtractedToolCallInformation:
        tools_raw = self._tools_content(model_output)
        if tools_raw is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = [
            ToolCall(
                id=m.group(1) or make_tool_call_id(),
                function=FunctionCall(
                    name=m.group(2).strip(),
                    arguments=m.group(3).strip(),
                ),
            )
            for m in self.COMPLETE_CALL_RE.finditer(tools_raw)
        ]

        if not tool_calls:
            # A tools channel opened but held no complete <call> (e.g. the model
            # hit max_tokens mid-call). Return the final text rather than
            # model_output, which still carries the tools-channel markers.
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=self._final_content(model_output),
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=self._final_content(model_output),
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> DeltaMessage | None:
        tools_raw = self._tools_content(current_text)
        if tools_raw is None:
            return None

        deltas: list[DeltaToolCall] = []

        # One uniform pass over every <call> opener, complete or not. The old
        # code ran a "complete calls" loop and a separate "partial call" branch
        # that measured argument length differently (.strip() vs raw, closed vs
        # greedy-to-end). Their disagreement corrupted _sent_args_chars: the
        # partial branch recorded a length that included "</arguments></call",
        # so once the call closed, args[sent:] was empty and the junk already
        # streamed to the client was never corrected.
        opens = list(self.CALL_OPEN_RE.finditer(tools_raw))

        for i, open_m in enumerate(opens):
            # This call's body runs to the next opener (or end of the channel).
            body_end = opens[i + 1].start() if i + 1 < len(opens) else len(tools_raw)
            body = tools_raw[open_m.end() : body_end]

            name_m = self.NAME_RE.search(body)
            if not name_m:
                # Name still being generated — emit nothing for this call yet.
                break
            name = name_m.group(1).strip()

            args = self._args_so_far(body[name_m.end() :]) or ""

            if i < len(self._sent_args_chars):
                # Already opened — send only the new argument chars.
                args_delta = args[self._sent_args_chars[i] :]
                if args_delta:
                    self._sent_args_chars[i] = len(args)
                    deltas.append(
                        DeltaToolCall(
                            index=i,
                            function=DeltaFunctionCall(arguments=args_delta),
                        )
                    )
            else:
                # First time seeing this call — open it with name + args so far.
                self._sent_args_chars.append(len(args))
                deltas.append(
                    DeltaToolCall(
                        index=i,
                        id=self._call_id(i, open_m.group(1)),
                        type="function",
                        function=DeltaFunctionCall(name=name, arguments=args or None),
                    )
                )

        return DeltaMessage(tool_calls=deltas) if deltas else None

    # ------------------------------------------------------------------
    # Request adjustment
    # ------------------------------------------------------------------

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        return request
