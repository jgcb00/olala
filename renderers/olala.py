"""Olala (Dragon 7A1B) renderer: a channel-structured chat template with XML tool calls.

Installed into a ``renderers`` checkout as ``renderers/olala.py`` by the env
setup script, which keeps that fork's diff to a single added file. The source of
truth is this file, in the olala repo -- it used to live in dragon-agentic,
where nothing else that serves or trains this model lived.

NOT the same thing as ``parsers/olala/`` beside it, despite parsing the same
channel format. Those are vLLM plugins (``ToolParserManager.register_module``,
``extract_tool_calls``) for the OpenAI server; this is a ``renderers``
``DefaultRenderer`` subclass (``_apply`` / ``parse_response`` /
``bridge_to_next_turn``) for verl/prime-rl rollouts. Neither substitutes for the
other.

The template wraps every message body in a named channel::

    <|im_start|>user<|channel_start|>text<|content|>...<|channel_end|><|im_end|>
    <|im_start|>tool<|channel_start|>metadata<|content|>{...}<|channel_end|>
                    <|channel_start|>tool_output<|content|>...<|channel_end|><|im_end|>

and an assistant turn is up to three channels — ``analysis`` (private
reasoning), ``final`` (the user-facing answer), ``tools`` (an XML
``<toolcalls>`` block). The generation prompt ends *inside* an open analysis
channel::

    <|im_start|>assistant<|channel_start|>analysis<|content|>

so a completion starts mid-channel: ``{analysis}<|channel_end|>
<|channel_start|>final<|content|>{answer}<|channel_end|><|im_end|>``.

Why this class exists at all, given ``DefaultRenderer`` renders the template
byte-correctly:

* ``parse_response`` — the channel delimiters are special tokens, so
  DefaultRenderer's ``_strip_special_tokens`` deletes them and concatenates the
  channels, yielding ``"{analysis}final{answer}"`` as ``content`` with
  ``reasoning_content=None``. Tool calls are XML, which no registered tool
  parser understands.
* ``bridge_to_next_turn`` — DefaultRenderer always returns ``None``, and the
  fallback full re-render *drops* earlier turns' ``reasoning_content`` unless
  ``preserve_thinking=True``, which rewrites history and breaks a token-level
  prefix chain.
* ``_apply`` — a ``tool`` message without ``metadata`` makes the checkpoint's
  template raise ``TypeError: Object of type Undefined is not JSON serializable``
  (``{{ message.metadata | tojson }}`` on a Jinja ``Undefined``). Harnesses emit
  bare ``{"role": "tool", "content": ...}``, so the renderer rebuilds it from the
  OpenAI fields (``{"name": ..., "id": tool_call_id}``) — the same rebuild the
  serving stack's template does (olala-vllm ``parsers/olala/chat_template.jinja``,
  2026-08-27), so training and serving render tool turns identically.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import ConfigDict
from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
    ToolSpec,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import BaseRendererConfig
from renderers.default import DefaultRenderer, _decode_tool_call_arguments

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CH_START = "<|channel_start|>"
CH_END = "<|channel_end|>"
CONTENT = "<|content|>"
ENDOFTEXT = "<|endoftext|>"

# One `<|channel_start|>NAME<|content|>BODY` unit. BODY runs to its close or to
# end-of-text, so a truncated final channel still parses.
_CHANNEL = re.compile(
    rf"{re.escape(CH_START)}(.*?){re.escape(CONTENT)}(.*?)(?:{re.escape(CH_END)}|\Z)",
    re.DOTALL,
)
_CALL = re.compile(r"<call\b([^>]*)>(.*?)</call>", re.DOTALL)
_CALL_OPEN = re.compile(r"<call\b[^>]*>")
_CALL_ID = re.compile(r'id\s*=\s*"([^"]*)"')
_CALL_NAME = re.compile(r"<name>(.*?)</name>", re.DOTALL)
_CALL_ARGS = re.compile(r"<arguments>(.*?)</arguments>", re.DOTALL)

# reasoning_effort values for which the template pre-CLOSES the analysis channel
# in the generation prompt, so the completion begins at `<|channel_start|>final`.
_NO_ANALYSIS = frozenset({"none", "disabled"})


class OlalaRendererConfig(BaseRendererConfig):
    """Config for :class:`OlalaRenderer`.

    ``extra="allow"`` so any other Jinja kwarg can still be forwarded, matching
    ``DefaultRendererConfig``'s escape hatch.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    name: Literal["olala"] = "olala"

    reasoning_effort: str = "medium"
    """Template kwarg; interpolated into the system block as "Reasoning effort:".
    ``"none"``/``"disabled"`` also make the generation prompt close the analysis
    channel immediately, so the model answers without reasoning."""

    preserve_thinking: bool = True
    """Template kwarg. Defaults to ``True``, unlike the template's own ``False``,
    because every dragon code path wants it:

    - a full re-render stays a byte-level EXTENSION of the previous turn's
      prompt+completion, so ``Engine``'s fallback re-render passes its prefix
      check instead of raising;
    - ``DefaultRenderer.render`` attributes tokens by re-rendering each message
      prefix, and with ``False`` an assistant turn's reasoning disappears once it
      is no longer ``loop.last``, making those per-message deltas negative.

    Set ``False`` only to reproduce the template's stock inference behaviour.
    """


class OlalaRenderer(DefaultRenderer):
    """Channel-aware renderer for the olala/Dragon family.

    ``render``/``render_ids`` are inherited: the checkpoint's Jinja template is
    the reference implementation and reproducing it by hand would only add a way
    to disagree with it. Everything the template cannot do — reading a completion
    back, and extending a turn without re-tokenizing sampled text — is here.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: OlalaRendererConfig | None = None,
    ):
        cfg = config or OlalaRendererConfig()
        self._tokenizer = tokenizer
        self.config = cfg
        # Parsing is hand-coded below; the pluggable parsers DefaultRenderer
        # consults would only fight it.
        self._tool_parser = None
        self._reasoning_parser = None
        # Implied "all": a bridge keeps the prior turn's sampled analysis tokens
        # verbatim, which IS retention across user-query boundaries.
        self.effective_thinking_retention = resolve_thinking_retention(cfg, "all")

        self._im_start = self._tok_id(IM_START)
        self._im_end = self._tok_id(IM_END)
        self._ch_start = self._tok_id(CH_START)
        self._ch_end = self._tok_id(CH_END)
        self._content = self._tok_id(CONTENT)
        self._endoftext = self._tok_id(ENDOFTEXT)

    def _tok_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == getattr(self._tokenizer, "unk_token_id", None):
            raise ValueError(f"tokenizer has no {token!r} — not an olala checkpoint?")
        return int(tid)

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    @property
    def supports_tools(self) -> bool:
        return True

    # ---- rendering -----------------------------------------------------
    def _apply(self, messages, *, tools=None, add_generation_prompt=False) -> list[int]:
        kwargs: dict[str, Any] = dict(self.config.model_extra or {})
        kwargs["reasoning_effort"] = self.config.reasoning_effort
        kwargs["preserve_thinking"] = self.config.preserve_thinking
        kwargs["add_generation_prompt"] = add_generation_prompt
        kwargs["tokenize"] = True
        kwargs["return_dict"] = False
        if tools is not None:
            kwargs["tools"] = tools
        # _decode_tool_call_arguments is DefaultRenderer's, and skipping it was a bug: the template
        # does `arguments | tojson`, so an OpenAI-format arguments STRING renders as a quoted JSON
        # string ("{\"cmd\":\"ls\"}") instead of an object. dragon's Engine emits exactly that shape.
        msgs = _decode_tool_call_arguments(_fill_tool_metadata(messages))
        return list(self._tokenizer.apply_chat_template(msgs, **kwargs))

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end]

    # ---- parsing -------------------------------------------------------
    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — arguments are JSON, so no schema needed
    ) -> ParsedResponse:
        ids = list(token_ids)
        while ids and ids[-1] in (self._im_end, self._endoftext):
            ids.pop()
        text = self._tokenizer.decode(ids, skip_special_tokens=False)

        reasoning: str | None = None
        content = ""
        tool_calls: list[ParsedToolCall] = []
        saw_final = False
        for name, body in _split_channels(text):
            name = name.strip()
            if name == "analysis":
                reasoning = body if reasoning is None else reasoning + body
            elif name == "final":
                content += body
                saw_final = True
            elif name == "tools":
                tool_calls.extend(_parse_tool_calls(body))
            # Unknown channel: dropped. It is neither the answer nor reasoning,
            # and guessing which would corrupt one of them.

        # A completion with no channel markup at all reads as "all analysis",
        # because the generation prompt opened that channel. For a checkpoint
        # that has not learned the format yet that would report an empty answer,
        # so treat an unstructured response as the answer instead.
        if not saw_final and CH_END not in text and reasoning:
            content, reasoning = reasoning, None

        return ParsedResponse(
            content=content,
            reasoning_content=reasoning or None,
            tool_calls=tool_calls,
        )

    # ---- bridging ------------------------------------------------------
    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — tools live in the turn-0 developer block
    ) -> RenderedTokens | None:
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None
        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        previous_ids = trim_to_turn_close(
            previous_prompt_ids, previous_completion_ids, {self._im_end, self._endoftext}
        )
        if previous_ids is None:
            # Truncated at max_tokens mid-channel. The canonical close needs BOTH
            # tokens (close the open channel, then the turn) — one is what
            # trim_to_turn_close's synthesize_close could give us, so do it here.
            previous_ids = (
                list(previous_prompt_ids) + list(previous_completion_ids)
                + [self._ch_end, self._im_end]
            )

        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []

        def emit(ids: list[int], msg_idx: int, is_content: bool) -> None:
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_content.extend([is_content] * len(ids))

        def emit_channel(name: str, body: str, msg_idx: int) -> None:
            # Every body is delimited by special tokens on both sides, so it
            # tokenizes identically alone and in context — no cross-boundary BPE
            # merge to preserve (unlike templates that use bare newlines).
            emit([self._ch_start] + self._encode(name) + [self._content], msg_idx, False)
            if body:
                emit(self._encode(body), msg_idx, True)
            emit([self._ch_end], msg_idx, False)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            if role == "user":
                emit([self._im_start] + self._encode("user"), i, False)
                emit_channel("text", content, i)
                emit([self._im_end], i, False)
            elif role == "tool":
                emit([self._im_start] + self._encode("tool"), i, False)
                emit_channel("metadata", json.dumps(_tool_metadata(msg)), i)
                emit_channel("tool_output", content, i)
                emit([self._im_end], i, False)
            else:
                # The template's message loop handles assistant/user/tool only:
                # a system or developer message here would render to nothing and
                # silently vanish from the prompt.
                return None

        emit([self._im_start] + self._encode("assistant"), -1, False)
        emit([self._ch_start] + self._encode("analysis") + [self._content], -1, False)
        if self.config.reasoning_effort.lower() in _NO_ANALYSIS:
            emit([self._ch_end], -1, False)

        total = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )


def _tool_metadata(msg: Message) -> dict:
    """A tool message's ``metadata`` channel body: explicit metadata wins, else it
    is rebuilt from the OpenAI fields — key order (name, id) matches the serving
    template's rebuild, so both render byte-identically."""
    if msg.get("metadata"):
        return msg["metadata"]
    meta: dict = {}
    if msg.get("name"):
        meta["name"] = msg["name"]
    if msg.get("tool_call_id"):
        meta["id"] = msg["tool_call_id"]
    return meta


def _fill_tool_metadata(messages: list[Message]) -> list[Message]:
    """Materialize ``metadata`` on tool messages (see ``_tool_metadata``); the
    checkpoint's template raises on a missing one (``| tojson`` on ``Undefined``)."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" and "metadata" not in m:
            m = {**m, "metadata": _tool_metadata(m)}
        out.append(m)
    return out


def _split_channels(text: str) -> list[tuple[str, str]]:
    """``[(channel_name, body)]`` for one assistant completion.

    The generation prompt leaves the analysis channel open, so any text before
    the first ``<|channel_start|>`` is that channel's body.
    """
    out: list[tuple[str, str]] = []
    if not text.startswith(CH_START):
        head, sep, rest = text.partition(CH_END)
        out.append(("analysis", head))
        if not sep:
            return out
        text = rest
    out.extend((m.group(1), m.group(2)) for m in _CHANNEL.finditer(text))
    return out


def _parse_tool_calls(body: str) -> list[ParsedToolCall]:
    """Parse a ``tools`` channel: ``<toolcalls><call id="..."><name>..</name>
    <arguments>{..}</arguments></call></toolcalls>``.

    Regex rather than an XML parser: a malformed block must yield a status, not
    an exception, and ``<arguments>`` holds arbitrary JSON that is not
    XML-escaped.
    """
    calls: list[ParsedToolCall] = []
    for m in _CALL.finditer(body):
        raw, attrs, inner = m.group(0), m.group(1), m.group(2)
        cid_m = _CALL_ID.search(attrs)
        cid = cid_m.group(1) if cid_m else None
        name_m = _CALL_NAME.search(inner)
        if name_m is None or not name_m.group(1).strip():
            calls.append(ParsedToolCall(raw=raw, id=cid, status=ToolCallParseStatus.MISSING_NAME))
            continue
        name = name_m.group(1).strip()
        args_m = _CALL_ARGS.search(inner)
        if args_m is None:
            calls.append(ParsedToolCall(raw=raw, name=name, id=cid,
                                        status=ToolCallParseStatus.MALFORMED_STRUCTURE))
            continue
        arg_text = args_m.group(1).strip()
        if not arg_text:
            calls.append(ParsedToolCall(raw=raw, name=name, id=cid, arguments={}))
            continue
        try:
            args = json.loads(arg_text)
        except ValueError:
            calls.append(ParsedToolCall(raw=raw, name=name, id=cid, arguments=arg_text,
                                        status=ToolCallParseStatus.INVALID_JSON))
            continue
        if not isinstance(args, dict):
            calls.append(ParsedToolCall(raw=raw, name=name, id=cid, arguments=args,
                                        status=ToolCallParseStatus.MALFORMED_STRUCTURE))
            continue
        calls.append(ParsedToolCall(raw=raw, name=name, id=cid, arguments=args))

    if not calls and _CALL_OPEN.search(body):
        # An opening <call> the model never closed — it hit max_tokens or a stop.
        calls.append(ParsedToolCall(raw=body, status=ToolCallParseStatus.UNCLOSED_BLOCK))
    elif not calls and "<toolcalls" in body:
        calls.append(ParsedToolCall(raw=body, status=ToolCallParseStatus.MALFORMED_STRUCTURE))
    return calls


def register() -> None:
    """Add ``olala`` to the renderer and config registries.

    ``_populate_registry`` short-circuits on a non-empty registry, so it must run
    BEFORE the insert — inserting first would leave every built-in renderer
    unregistered.
    """
    from renderers import base, configs

    base._populate_registry()
    base.RENDERER_REGISTRY.setdefault("olala", OlalaRenderer)
    configs._CONFIG_BY_NAME.setdefault("olala", OlalaRendererConfig)


register()
