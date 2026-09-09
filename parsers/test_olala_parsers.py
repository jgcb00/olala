#!/usr/bin/env python
"""Regression tests for parsers/olala/ (reasoning + tool-call parsers).

Offline mode replays vLLM's own parse() / streaming chain over canned model
output -- no GPU, no server, deterministic:

    .venv/bin/python test_olala_parsers.py

Live mode additionally drives a running server (both streaming and
non-streaming) and validates the tool calls it returns:

    .venv/bin/python test_olala_parsers.py --live            # :8010
    .venv/bin/python test_olala_parsers.py --live --port 8020

NB the server loads the parsers at startup, so restart it after editing
parsers/olala/ or --live will still exercise the old code.
"""

import argparse
import json
import os
import pathlib
import sys
import types
import urllib.error
import urllib.request

# Repo root (…/olala), derived from this file so the suite travels with it.
BASE = str(pathlib.Path(__file__).resolve().parent.parent)
# A checkpoint to read the chat template and tokenizer from. Override with
# OLALA_CKPT; the default is where convert/ is documented to put an export.
CKPT = os.environ.get("OLALA_CKPT") or (
    "/data/home/gaetan.caillaut/dragon-sft/7A1B/training/checkpoints/"
    "65k-betterpacks-lrfix/huggingface/iter_0060000"
)

import os

# OLALA_PARSERS_DIR lets you point the offline tests at another copy of the
# parsers (e.g. to reproduce a bug against a pre-fix version).
sys.path.insert(0, os.environ.get("OLALA_PARSERS_DIR", f"{BASE}/parsers/olala"))
sys.path.insert(0, f"{BASE}/vllm-v026")

# Markers that must never reach a client.
MARKERS = (
    "<|channel_start|>",
    "<|channel_end|>",
    "<|content|>",
    "<toolcalls>",
    "<call ",
    "</arguments>",
    "</call>",
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

failures: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("   PASS  " if cond else "   FAIL  ") + label)
    if not cond:
        failures.append(label)


def valid_json(s):
    try:
        return json.loads(s), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


# ======================================================================
# Offline: replay vLLM's parser chain
# ======================================================================


def offline() -> None:
    import olala_reasoning_parser
    import olala_tool_parser
    from transformers import AutoTokenizer
    from vllm.parser.abstract_parser import DelegatingParser

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

    class Combined(DelegatingParser):
        """What ParserManager.get_parser() builds for --reasoning-parser olala
        --tool-call-parser olala."""

        reasoning_parser_cls = olala_reasoning_parser.OlalaParser
        tool_parser_cls = olala_tool_parser.OlalaToolParser

    def req(tools=(WEATHER_TOOL,), tool_choice="auto"):
        r = types.SimpleNamespace()
        r.tools = list(tools) if tools else None
        r.tool_choice = tool_choice
        return r

    def nonstream(raw, request):
        return Combined(tok, request.tools).parse(
            raw, request, enable_auto_tools=True
        )

    def stream(raw):
        """Feed the tool parser one token at a time and accumulate deltas the
        way an OpenAI client does."""
        tp = olala_tool_parser.OlalaToolParser(tok)
        ids = tok.encode(raw, add_special_tokens=False)
        calls: dict[int, dict] = {}
        prev_text, prev_ids = "", []
        for k in range(len(ids)):
            cur_ids = ids[: k + 1]
            cur_text = tok.decode(cur_ids, skip_special_tokens=False)
            dm = tp.extract_tool_calls_streaming(
                prev_text,
                cur_text,
                cur_text[len(prev_text) :],
                prev_ids,
                cur_ids,
                [ids[k]],
                None,
            )
            if dm and dm.tool_calls:
                for tc in dm.tool_calls:
                    slot = calls.setdefault(
                        tc.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
            prev_text, prev_ids = cur_text, cur_ids
        return calls

    def case(label, raw, *, calls, content, request=None, cmp_stream=True):
        request = request or req()
        print(f"\n--- {label}")
        reasoning, got_content, tool_calls = nonstream(raw, request)
        tool_calls = tool_calls or []

        check(len(tool_calls) == calls, f"non-stream: {calls} call(s), got {len(tool_calls)}")
        check(got_content == content, f"non-stream: content == {content!r}, got {got_content!r}")
        for tc in tool_calls:
            _, err = valid_json(tc.arguments)
            check(err is None, f"non-stream: {tc.name} args valid JSON ({tc.arguments!r})")
        for field, val in (("content", got_content), ("reasoning", reasoning)):
            leaked = [m for m in MARKERS if val and m in val]
            check(not leaked, f"non-stream: no markers in {field} (found {leaked})")

        if not cmp_stream:
            return
        sc = stream(raw)
        check(len(sc) == calls, f"stream: {calls} call(s), got {len(sc)}")
        for i in sorted(sc):
            _, err = valid_json(sc[i]["arguments"])
            check(err is None, f"stream: call[{i}] args valid JSON ({sc[i]['arguments']!r})")
        ns = {i: (tc.name, valid_json(tc.arguments)[0]) for i, tc in enumerate(tool_calls)}
        st = {i: (sc[i]["name"], valid_json(sc[i]["arguments"])[0]) for i in sorted(sc)}
        check(ns == st, f"stream == non-stream ({ns} vs {st})")

    A = "Some analysis here."
    FINAL_EMPTY = "<|channel_start|>final<|content|><|channel_end|>"

    def tools_ch(*calls):
        return (
            "<|channel_start|>tools<|content|><toolcalls>"
            + "".join(calls)
            + "</toolcalls><|channel_end|>"
        )

    def call(cid, name, args):
        return f'<call id="{cid}"><name>{name}</name><arguments>{args}</arguments></call>'

    print("=" * 70)
    print("OFFLINE")
    print("=" * 70)

    case(
        "single tool call, empty final channel",
        f"{A}<|channel_end|>{FINAL_EMPTY}" + tools_ch(call("call_0", "get_weather", '{"city": "Paris"}')),
        calls=1,
        content=None,
    )

    case(
        "two parallel tool calls",
        f"{A}<|channel_end|>{FINAL_EMPTY}"
        + tools_ch(
            call("call_0", "get_weather", '{"city": "Paris"}'),
            call("call_1", "get_weather", '{"city": "Tokyo", "unit": "celsius"}'),
        ),
        calls=2,
        content=None,
    )

    case(
        "final text alongside a tool call",
        f"{A}<|channel_end|><|channel_start|>final<|content|>Let me check that.<|channel_end|>"
        + tools_ch(call("call_0", "get_weather", '{"city": "Paris"}')),
        calls=1,
        content="Let me check that.",
    )

    case(
        "plain answer, no tools channel",
        f"{A}<|channel_end|><|channel_start|>final<|content|>It is sunny in Paris.<|channel_end|>",
        calls=0,
        content="It is sunny in Paris.",
    )

    case(
        "nested JSON and </arg>-like text in arguments",
        f"{A}<|channel_end|>{FINAL_EMPTY}"
        + tools_ch(
            call(
                "call_0",
                "get_weather",
                '{"city": "Paris", "note": "a </arg> like string", "n": {"deep": [1,2]}}',
            )
        ),
        calls=1,
        content=None,
    )

    case(
        "truncated mid-call (hit max_tokens)",
        f"{A}<|channel_end|>{FINAL_EMPTY}"
        '<|channel_start|>tools<|content|><toolcalls>'
        '<call id="call_0"><name>get_weather</name><arguments>{"city": "Par',
        calls=0,
        content=None,
        cmp_stream=False,
    )

    case(
        "tool_choice='none' must not leak the tools channel",
        f"{A}<|channel_end|><|channel_start|>final<|content|>Sure.<|channel_end|>"
        + tools_ch(call("call_0", "get_weather", '{"city": "Paris"}')),
        calls=0,
        content="Sure.",
        request=req(tool_choice="none"),
        cmp_stream=False,
    )

    case(
        "no tools in the request",
        f"{A}<|channel_end|><|channel_start|>final<|content|>Hello.<|channel_end|>",
        calls=0,
        content="Hello.",
        request=req(tools=None),
        cmp_stream=False,
    )

    # tool_choice="required" must route through auto parsing (this format is XML,
    # so vLLM's JSON-based required/named handling cannot read it).
    case(
        "tool_choice='required' falls back to auto parsing",
        f"{A}<|channel_end|>{FINAL_EMPTY}"
        + tools_ch(call("call_0", "get_weather", '{"city": "Paris"}')),
        calls=1,
        content=None,
        request=req(tool_choice="required"),
        cmp_stream=False,
    )

    # Named tool choice, using the real request type so isinstance() checks fire.
    print("\n--- tool_choice=<named> falls back to auto parsing")
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionNamedToolChoiceParam,
    )

    named = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "get_weather"}}
    )
    r = req(tool_choice=named)
    _, got_content, tool_calls = nonstream(
        f"{A}<|channel_end|>{FINAL_EMPTY}"
        + tools_ch(call("call_0", "get_weather", '{"city": "Paris"}')),
        r,
    )
    tool_calls = tool_calls or []
    check(len(tool_calls) == 1, f"non-stream: 1 call, got {len(tool_calls)}")
    if tool_calls:
        args, err = valid_json(tool_calls[0].arguments)
        check(err is None, f"non-stream: args valid JSON ({tool_calls[0].arguments!r})")
        check(args == {"city": "Paris"}, f"non-stream: args == city/Paris, got {args!r}")


# ======================================================================
# Live: drive a running server
# ======================================================================


def live(port: int) -> None:
    base = f"http://localhost:{port}/v1"

    def post(payload, stream=False, timeout=1800):
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            # Surface the server's message; a bare "HTTP Error 400" says nothing.
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail)["error"]["message"]
            except Exception:  # noqa: BLE001
                detail = detail[:500]
            check(False, f"HTTP {e.code} from server: {detail}")
            return None

    def body(messages, tools=(WEATHER_TOOL,), tool_choice="auto", **kw):
        p = {
            "model": "olala-7a1b",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 700,
        }
        if tools:
            p["tools"] = list(tools)
            p["tool_choice"] = tool_choice
        p.update(kw)
        return p

    print("\n" + "=" * 70)
    print(f"LIVE  (:{port})")
    print("=" * 70)

    # -- non-streaming ------------------------------------------------
    print("\n--- non-streaming: single tool call")
    r = post(body([{"role": "user", "content": "What's the weather in Paris right now?"}]))
    choice = json.loads(r.read())["choices"][0] if r else {}
    msg = choice.get("message") or {}
    tcs = msg.get("tool_calls") or []
    check(bool(tcs), f"tool_calls present (finish_reason={choice.get('finish_reason')!r})")
    for tc in tcs:
        print(f"       {tc['id']} {tc['function']['name']} {tc['function']['arguments']!r}")
        _, err = valid_json(tc["function"]["arguments"])
        check(err is None, f"arguments valid JSON ({err})")
        check(tc["function"]["name"] == "get_weather", "name == get_weather")
    for f in ("content", "reasoning"):
        v = msg.get(f)
        leaked = [m for m in MARKERS if v and m in v]
        check(not leaked, f"no markers in {f} (found {leaked})")

    # -- streaming ----------------------------------------------------
    print("\n--- streaming: single tool call")
    calls: dict[int, dict] = {}
    content_acc, finish = "", None
    sr = post(body([{"role": "user", "content": "What's the weather in Paris right now?"}], stream=True))
    for r in ([sr] if sr else []):
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            ch = json.loads(line[6:])["choices"][0]
            d = ch.get("delta") or {}
            if d.get("content"):
                content_acc += d["content"]
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]

    check(bool(calls), f"tool_calls streamed (finish_reason={finish!r})")
    for i in sorted(calls):
        c = calls[i]
        print(f"       [{i}] {c['id']} {c['name']} {c['arguments']!r}")
        _, err = valid_json(c["arguments"])
        check(err is None, f"stream call[{i}] arguments valid JSON ({err})")
    leaked = [m for m in MARKERS if content_acc and m in content_acc]
    check(not leaked, f"no markers in streamed content (found {leaked})")

    # -- parallel tool calls, both modes ------------------------------
    # The offline suite proves the PARSER splits parallel calls, but its input is
    # hand-written markup. This asks the MODEL to emit them. NB the instruction
    # has to be explicit: "compare X and Y" phrasing gets one call and a
    # serialized follow-up, which is model behaviour, not a parsing failure.
    parallel_prompt = (
        "I need the weather for BOTH Paris and Tokyo. "
        "Call the tool once per city, in the same turn."
    )

    def collect(stream):
        """Return {index: {id, name, arguments}} from either response mode."""
        got: dict[int, dict] = {}
        r = post(body([{"role": "user", "content": parallel_prompt}], stream=stream))
        if not r:
            return got, None
        if not stream:
            ch = json.loads(r.read())["choices"][0]
            for i, tc in enumerate(ch["message"].get("tool_calls") or []):
                got[i] = {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            return got, ch.get("finish_reason")
        fin = None
        with r as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    break
                ch = json.loads(line[6:])["choices"][0]
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    slot = got.setdefault(
                        tc["index"], {"id": None, "name": None, "arguments": ""}
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    fin = ch["finish_reason"]
        return got, fin

    for mode, is_stream in (("non-streaming", False), ("streaming", True)):
        print(f"\n--- {mode}: parallel tool calls")
        got, fin = collect(is_stream)
        for i in sorted(got):
            c = got[i]
            print(f"       [{i}] {c['id']} {c['name']} {c['arguments']!r}")
        check(
            len(got) >= 2,
            f"{mode}: model emitted >=2 calls (got {len(got)}, finish_reason={fin!r})",
        )
        cities = set()
        for i in sorted(got):
            args, err = valid_json(got[i]["arguments"])
            check(err is None, f"{mode}: call[{i}] arguments valid JSON ({err})")
            if isinstance(args, dict) and args.get("city"):
                cities.add(args["city"].lower())
        ids = [got[i]["id"] for i in sorted(got)]
        # Uniqueness is what matters: an agent echoes tool_call_id back, so a
        # collision would misroute results. The model is NOT consistent about
        # format -- it emits e.g. id="call_0" then id="1" in one turn -- which is
        # ugly but harmless as long as they stay distinct.
        check(
            len(ids) == len(set(ids)) and all(ids),
            f"{mode}: call ids present and unique ({ids})",
        )
        if len(got) >= 2:
            check(
                len(cities) >= 2,
                f"{mode}: calls cover distinct cities ({sorted(cities)})",
            )

    # -- round trip: feed the tool result back ------------------------
    print("\n--- round trip: tool result -> final answer")
    if calls:
        c = calls[0]
        msgs = [
            {"role": "user", "content": "What's the weather in Paris right now?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": c["id"],
                "content": json.dumps({"city": "Paris", "temp_c": 21, "sky": "clear"}),
                "metadata": {"tool": c["name"]},
            },
        ]
        r = post(body(msgs))
        msg = json.loads(r.read())["choices"][0]["message"] if r else {}
        print(f"       content: {(msg.get('content') or '')[:200]!r}")
        check(bool(msg.get("content")), "final answer has content")
        check("21" in (msg.get("content") or ""), "final answer mentions the tool result (21)")
        leaked = [m for m in MARKERS if msg.get("content") and m in msg["content"]]
        check(not leaked, f"no markers in final content (found {leaked})")

    # -- no tools: plain chat still works ------------------------------
    print("\n--- no tools: plain chat")
    r = post(body([{"role": "user", "content": "Say hello in one short sentence."}], tools=None))
    msg = json.loads(r.read())["choices"][0]["message"] if r else {}
    print(f"       content: {(msg.get('content') or '')[:160]!r}")
    check(bool(msg.get("content")), "content present")
    leaked = [m for m in MARKERS if msg.get("content") and m in msg["content"]]
    check(not leaked, f"no markers in content (found {leaked})")


# ======================================================================
# Chat template: the "tool" branch must render for every field shape
# ======================================================================


def template() -> None:
    """The checkpoint's template renders {{ message.metadata | tojson }}
    unguarded. `metadata` is not an OpenAI field, so vLLM drops it and Jinja
    raised "Object of type Undefined is not JSON serializable" -> HTTP 400 on
    every tool-result turn. The tokenizer's template fixes that; these
    cases pin it."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    # The template is no longer shipped beside the parsers: the fix below is
    # upstream in the tokenizer (channels-v4 onward), so every export carries
    # it. Read the one the checkpoint actually ships.
    tmpl = open(f"{CKPT}/chat_template.jinja").read()

    user = {"role": "user", "content": "Weather in Paris?"}
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    result = '{"temp_c": 21}'

    shapes = {
        "OpenAI standard (tool_call_id only)": {
            "role": "tool", "tool_call_id": "call_0", "content": result,
        },
        "OpenAI + name": {
            "role": "tool", "tool_call_id": "call_0", "name": "get_weather",
            "content": result,
        },
        "explicit metadata (training/test shape)": {
            "role": "tool", "metadata": {"name": "get_weather", "id": "call_0"},
            "content": result,
        },
        "bare role+content (shape in the SFT data)": {
            "role": "tool", "content": result,
        },
    }

    print("\n" + "=" * 70)
    print("CHAT TEMPLATE")
    print("=" * 70)

    for label, tool_msg in shapes.items():
        print(f"\n--- {label}")
        try:
            out = tok.apply_chat_template(
                [user, assistant, tool_msg],
                tools=[WEATHER_TOOL],
                chat_template=tmpl,
                add_generation_prompt=True,
                tokenize=False,
            )
            i = out.find("<|im_start|>tool")
            check(i != -1, "tool turn rendered")
            print(f"       {out[i : out.find('<|im_end|>', i)]!r}")
        except Exception as e:  # noqa: BLE001
            check(False, f"renders without error ({type(e).__name__}: {e})")

    # tokenizer_tests.py invariant: adding a tool turn must not perturb the
    # prefix produced by the turns before it.
    print("\n--- prefix-preserving (7A1B/tokenizer_tests.py invariant)")

    def ids(msgs, **kw):
        text = tok.apply_chat_template(
            msgs, tools=[WEATHER_TOOL], chat_template=tmpl, tokenize=False, **kw
        )
        return tok.encode(text, add_special_tokens=False)

    before = ids([user, assistant])
    after = ids(
        [user, assistant, {"role": "tool", "tool_call_id": "call_0", "content": result}],
        add_generation_prompt=True,
    )
    check(after[: len(before)] == before, "tool turn does not perturb the prefix")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also test a running server")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--skip-offline", action="store_true")
    args = ap.parse_args()

    if not args.skip_offline:
        offline()
        template()
    if args.live:
        live(args.port)

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
