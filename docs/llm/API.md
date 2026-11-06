# Task brief: connect a chat client or agent

> **For your frontier model.** People: read [the API guide](../API.md) instead, and see
> [how to use this brief](README.md). Model: also follow [AGENTS.md](../../AGENTS.md).
> If you cannot open a linked file, ask for it; do not guess.

Read [the API guide](../API.md) and `tools/api-check.py` first. Identify which
machine runs the client's HTTP backend. Default model endpoint:
`http://127.0.0.1:7710/v1`; console: port 7730. The model port is the head's
configured control port + 10. Wait for Ready; GET `/v1/models` supplies the ID.

Try model discovery and one short Chat Completions request from the same
environment as the client. Use `think:false` on the native Metal head, and
`chat_template_kwargs: {"enable_thinking": false}` on llama.cpp Vulkan, with
`max_tokens:64` and a harmless prompt. Supplying both is convenient for this
example. The v0.3.0 bundled API checker can pass on Vulkan while emitting
reasoning tokens; do not equate content being present with thinking disabled.
Then exercise SSE and its final usage;
do not count chunks as tokens. `tools/api-check.py --tool-test` additionally
checks a validated arithmetic function call, its tool result and follow-up.

The local API is unauthenticated and loopback-only. `fronk-local` is a dummy
key for clients that require a field; the cluster pairing code is unrelated.
Do not publish the engine port or change firewall, tunnel or authentication
settings without understanding the user's access requirements. Nodes need no
SSH. An optional tunnel for a separate API client is a different concern.

Do not promise all OpenAI/Anthropic/Responses fields work because the server
has a compatible endpoint. The pinned parser skips some extension fields;
forced tool selection is not enforced for Chat Completions. Use automatic tool
selection, validate tool arguments in the client and keep execution in its
normal approval/sandbox system. Never execute arbitrary model-proposed code
as an API connectivity check.

If curl works but the application does not, inspect its base URL composition,
model ID, request JSON, streaming, timeout, container network and browser origin.
Keep a minimal reproduction and identify the failing layer before changing
Fronk. Run the same reproduction after a fix, then document the client/version
and tested features. Do not include keys or private prompts in a report.
