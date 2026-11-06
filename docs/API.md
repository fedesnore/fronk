# Use your own chat interface or agent

Fronk's web chat is optional. Once the head console says **Ready**, an ordinary
HTTP client can use the same distributed model. It does not need to know where
the slices live, and it does not talk to the nodes.

## Connect a client running on the head

In an interface that accepts a custom **OpenAI-compatible Chat Completions**
provider, enter:

| Setting | Value with the default Fronk configuration |
|---|---|
| Provider | OpenAI-compatible / custom endpoint |
| Base URL | `http://127.0.0.1:7710/v1` |
| Model | `qwen3.8-flash-next` — confirm with `/v1/models` below |
| API key | Not required by the local server; use `fronk-local` if the client requires a nonempty field |
| Streaming | Enabled |
| Request timeout | At least 300 seconds while trying longer prompts |

Do not enter the cluster pairing code as an API key. The local model API has
no authentication; the placeholder above is not a password. It binds to
loopback, so other computers cannot reach it directly.

Some clients ask for a complete endpoint instead of a base URL. In that case
use `http://127.0.0.1:7710/v1/chat/completions`. Avoid adding `/v1` twice.
The API port is `control-port + 10` in the head's `fronk.conf`; the defaults
are 7700 and 7710. Port 7730 serves the Fronk console, not this provider API.

First check model discovery on the head:

```sh
curl --fail --silent --show-error http://127.0.0.1:7710/v1/models
```

Select the returned `data[0].id`. The tested Metal head returns
`qwen3.8-flash-next`; a different head adapter may use another identifier.

Then ask for a short answer:

```sh
curl --fail --silent --show-error http://127.0.0.1:7710/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","think":false,"chat_template_kwargs":{"enable_thinking":false},"max_tokens":64,"messages":[{"role":"user","content":"Explain what a network is in one sentence."}]}'
```

For streaming, add `"stream":true` and `"stream_options":{"include_usage":true}`
to the JSON and `--no-buffer` to curl. Content arrives in
`choices[].delta.content`; the final usage event reports token counts, followed
by `data: [DONE]`. An SSE fragment is not necessarily one token.

The native Metal adapter accepts `think:false` to disable reasoning for a quick
first test. The llama.cpp Vulkan adapter instead uses
`chat_template_kwargs: {"enable_thinking": false}`. The example supplies both
fields; put the appropriate field in your client's extra request parameters.
The v0.3.0 bundled API-check example supplies only `think:false`: it can pass
on the Vulkan head while still generating reasoning tokens. A passing short
answer does not prove reasoning was disabled. Do not
assume that every field supported by a hosted provider has the same meaning
here. This guide qualifies Chat Completions, not every provider API.

## A small, working agent example

[tools/api-check.py](../tools/api-check.py) is a complete Python example without
third-party dependencies. Run it on the head after Ready:

```sh
python3 ~/fronk/src/tools/api-check.py --tool-test
```

It discovers the model, completes an ordinary request, reads a streaming
response and checks its final token usage. Then it offers one tool,
`add_numbers`, to the model. The model proposes its name and JSON arguments;
the client validates them, performs a fixed local addition and sends a `tool`
message with the matching `tool_call_id`. The model uses that result in its
next answer. Fronk never executes the proposed tool itself.

This full cycle was exercised on the four-computer Metal-head cluster:
the model requested `17 + 25`, the client returned 42 and the final answer used
42. JSON completion, text streaming, model discovery and rejection of a
malformed request also passed. The repeatable example matters more than a
claim that every chat application or agent framework has been tested.

Choose Chat Completions and automatic tool selection in your agent. The pinned
adapter accepts tool schemas and tool results. Do not rely on forced tool
selection: the adapter's parser does not enforce the full hosted-provider
contract. Start with a harmless tool and one conversation at a time. A long
agent prompt still has to be read by the distributed model; decode tok/s does
not describe that initial wait.

## A client on another computer

`127.0.0.1` means the computer running the **client's HTTP backend**. For a
desktop application on the head, that is already correct. For a web UI with a
separate server or a container, it may refer to that server or container
instead. Changing only the address in a browser cannot fix this distinction.

Keep the model API private. One option for a trusted remote client is a local
SSH tunnel, if you already use SSH on the head:

```sh
ssh -N -L 7710:127.0.0.1:7710 HEAD_USER@HEAD_ADDRESS
```

Run that on the client computer, then use the same base URL there. This is an
optional way to reach your own API; SSH is not used for Fronk installation,
node discovery or inference. An authenticated private reverse proxy is another
option. Do not expose the unauthenticated engine port to the public Internet.
Container networking varies by platform; first make the curl check work from
the environment that actually sends the request.

## If a connection fails

| What happens | Check first |
|---|---|
| Connection refused | The head must finish Load, Check and Test and reach Ready. Check the port. |
| Model not found | Use the ID returned by `/v1/models`, not a downloaded filename. |
| 404 | Check whether the client wants a base URL or the full endpoint. |
| No visible text | Disable thinking for the first test; allow time to read the prompt. |
| Stream seems buffered | Check the client's streaming setting and any proxy buffering. |
| A tool never runs | Inspect `tool_calls`; the client, not Fronk, executes tools and returns their results. |
| Works in curl, fails in a browser UI | Check where that UI's backend runs and its origin/CORS requirements. Do not disable browser protections globally. |

For help, give a frontier model [the API brief](llm/API.md), your client name and the exact
error. Share a harmless minimal request, not your private conversation or keys.
