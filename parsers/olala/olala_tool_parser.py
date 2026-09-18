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
        # Chars of final-channel content already streamed (see _content_delta).
        self._sent_content_chars = 0
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

    def _final_content_so_far(self, text: str) -> str:
        """The user-facing slice of `text`, markers removed.

        Same two input shapes as _final_content(), but monotonic and unstripped:
        the result only ever grows as `text` does, so the streaming path can
        diff it against what it has already sent.
        """
        if self.FINAL_MARKER in text:
            text = text.split(self.FINAL_MARKER, 1)[1]
        # The final channel is closed by <|channel_end|>; the tools channel and
        # anything after it are never user-facing text.
        for stop in (self.CHANNEL_END, self.TOOLS_MARKER):
            cut = text.find(stop)
            if cut != -1:
                text = text[:cut]
        return text

    def _content_delta(self, current_text: str) -> str:
        """New final-channel chars since the last delta."""
        content = self._final_content_so_far(current_text)
        if self._sent_content_chars == 0:
            # Non-streaming .strip()s its content. Leading whitespace can be
            # dropped the same way here; trailing whitespace cannot, since it is
            # only knowable as trailing once the channel closes.
            delta = content.lstrip()
        else:
            delta = content[self._sent_content_chars :]
        self._sent_content_chars = len(content)
        return delta

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
        # Content first. Once the reasoning parser reports reasoning-end -- which
        # for this format is the START of the final channel, not its end -- vLLM
        # stops calling it and hands this parser the whole remaining stream
        # (DelegatingParser.parse_delta). Emitting only tool calls therefore
        # dropped every final-channel token. The shipped parsers (hermes et al.)
        # return content from here for exactly this reason.
        content_delta = self._content_delta(current_text)

        tools_raw = self._tools_content(current_text)
        deltas: list[DeltaToolCall] = []

        # One uniform pass over every <call> opener, complete or not. The old
        # code ran a "complete calls" loop and a separate "partial call" branch
        # that measured argument length differently (.strip() vs raw, closed vs
        # greedy-to-end). Their disagreement corrupted _sent_args_chars: the
        # partial branch recorded a length that included "</arguments></call",
        # so once the call closed, args[sent:] was empty and the junk already
        # streamed to the client was never corrected.
        opens = list(self.CALL_OPEN_RE.finditer(tools_raw)) if tools_raw else []

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

        if not content_delta and not deltas:
            return None
        return DeltaMessage(content=content_delta or None, tool_calls=deltas)

    # ------------------------------------------------------------------
    # Request adjustment
    # ------------------------------------------------------------------

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False

        # ChatCompletionRequest defaults tool_choice to "none", so every request
        # that sends no tools arrives here as "none" -- and DelegatingParser.
        # _extract_tool_calls_streaming() short-circuits that case with
        #     return (DeltaMessage(content=delta_text) if delta_text else None)
        # without consulting either parser. Since reasoning-end is reported at
        # the START of the final channel (see OlalaParser.is_reasoning_end), the
        # reasoning parser has already stopped by then too, so nothing strips
        # the channel markers and <|channel_end|> reaches the client inside
        # `content`.
        #
        # Promoting to "auto" routes the stream back through this parser, which
        # emits clean content. Confined to requests that declared no tools, so
        # it cannot override a caller who deliberately disabled tool calling:
        # the model has no tool list to call from, and the tools channel stays
        # empty. A request that sends tools AND sets tool_choice="none"
        # explicitly still takes the bypass and still leaks; that shape needs
        # the fix in vLLM itself.
        if not getattr(request, "tools", None) and (
            getattr(request, "tool_choice", None) == "none"
        ):
            request.tool_choice = "auto"
        return request
