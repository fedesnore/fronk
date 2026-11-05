#!/usr/bin/env python3
"""Exercise a running head through its public HTTP API, using only Python.

Run after the console says Ready. The optional tool test executes one local,
fixed arithmetic function; it never executes a command proposed by the model.
This is an integration example and a live check, not part of a normal build.
"""
import argparse
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:7710/v1")
    parser.add_argument("--tool-test", action="store_true")
    parser.add_argument("--output", help="Save the harmless test results as JSON")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def request(path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    result = {}
    try:
        with request("/models") as response:
            model = json.load(response)["data"][0]["id"]
        result["model"] = model
        body = {"model": model, "think": False, "temperature": 0,
                "max_tokens": 64, "messages": [{"role": "user", "content": "Reply with one short sentence explaining what a network is."}]}
        with request("/chat/completions", body) as response:
            completed = json.load(response)
        assert completed["choices"][0]["message"]["content"], "empty non-streaming answer"
        assert completed["usage"]["completion_tokens"] > 0, "missing completion token count"
        result["non_streaming"] = completed
        body.update(stream=True, stream_options={"include_usage": True})
        pieces, usage, finished, first, last = [], None, False, None, None
        started = time.monotonic()
        with request("/chat/completions", body) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    finished = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise ValueError(event["error"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    piece = choice.get("delta", {}).get("content", "")
                    if piece:
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                        pieces.append(piece)
        assert finished and pieces and usage and usage["completion_tokens"] > 0, "incomplete SSE response"
        result["streaming"] = {"text": "".join(pieces), "usage": usage,
                               "sse_fragments": len(pieces), "time_to_first_s": first - started,
                               "request_s": time.monotonic() - started,
                               "first_to_last_content_s": last - first}
        # Do not treat fragment count as token count: one SSE chunk can contain
        # more than one token. Keep both observations for a reviewer.
        if args.tool_test:
            messages = [{"role": "user", "content": "Use the add_numbers tool to add 17 and 25. Do not calculate it yourself. After the tool replies, state its result in one short sentence."}]
            tool = {"type": "function", "function": {"name": "add_numbers", "description": "Add two integers.",
                    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
            body.update(stream=False, messages=messages, tools=[tool], tool_choice="auto", max_tokens=128)
            with request("/chat/completions", body) as response:
                proposal = json.load(response)
            assistant = proposal["choices"][0]["message"]
            calls = assistant.get("tool_calls", [])
            assert len(calls) == 1 and calls[0]["function"]["name"] == "add_numbers", "model did not propose the expected tool"
            params = json.loads(calls[0]["function"]["arguments"])
            assert set(params) == {"a", "b"} and all(type(v) is int for v in params.values()), "unexpected tool arguments"
            assert sorted(params.values()) == [17, 25], "unexpected arithmetic inputs"
            value = params["a"] + params["b"]
            messages.extend([assistant, {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps({"result": value})}])
            body["tool_choice"] = "none"
            with request("/chat/completions", body) as response:
                answer = json.load(response)
            assert "42" in answer["choices"][0]["message"]["content"], "tool result was not used"
            result["tool_round_trip"] = {"proposal": proposal, "answer": answer, "local_result": value}
        # Bad input should produce a structured client error, not a stalled run.
        try:
            request("/chat/completions", {"model": model, "messages": "not an array"})
            raise AssertionError("malformed messages were accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 400, f"unexpected error status {error.code}"
            result["invalid_request_status"] = error.code
        result["passed"] = True
    except (OSError, ValueError, KeyError, IndexError, AssertionError) as error:
        parser.exit(1, f"API check failed: {error}\nWait for Ready on the head and check the API address.\n")
    if args.output:
        with open(args.output, "x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(f"PASS: {model}; model discovery, JSON completion, SSE + usage, invalid request" +
          (", tool call + local result + follow-up" if args.tool_test else ""))


if __name__ == "__main__":
    main()
