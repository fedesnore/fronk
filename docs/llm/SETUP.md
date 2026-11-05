# Set up Fronk on my computers

> **For your coding agent.** People: open Codex or Claude Code on the computer with the
> best GPU, paste this whole file, and do what it asks. You do not need to know networking.
> Model: also follow [AGENTS.md](../../AGENTS.md).

You are setting up a complete Fronk cluster for someone who may know nothing about
networks or terminals. Take it from nothing to a finished chat. Do the technical work
yourself and ask only for what needs their hands or their decision.

Fronk runs one large model across several computers: one **head** with a GPU, and
**nodes** that hold slices of the model.

## How to work with this person

- Use plain words. Say "the cable between the Mac and the iMac", not "the interface".
- Give one physical action at a time, name the computer, and say how they will know it
  worked. Check the result yourself when you can.
- Before changing a computer's network settings or installing anything, say in one
  sentence what changes and how to undo it, and wait for a yes.
- Keep going until a real chat answer appears, or name exactly what blocks it.

## 1. Get the files

Work on the head. If this computer has no Fronk folder, get the release:

```sh
mkdir -p ~/fronk-download && cd ~/fronk-download
curl -fLO https://github.com/fedesnore/fronk/releases/download/v0.3.0/fronk-manager.sh
curl -fLO https://github.com/fedesnore/fronk/releases/download/v0.3.0/SHA256SUMS
grep ' fronk-manager.sh$' SHA256SUMS | shasum -a 256 -c
```

Then read its guides without installing yet:

```sh
line=$(awk '/^__FRONK_PAYLOAD__$/ {print NR + 1; exit}' fronk-manager.sh)
mkdir -p source && tail -n +"$line" fronk-manager.sh | tar -xzf - -C source
```

The briefs are in `source/fronk/docs/llm/`. Read `AGENTS.md`, `docs/llm/NETWORK.md` and
`docs/llm/INSTALL.md` there. If a download fails, ask the person to paste those files.

## 2. Learn what they have

Ask only this, in one message:

1. Which computers do you want to use? A rough description is enough.
2. Which one is the head? Suggest the Mac with Apple Silicon. Other heads are experimental.
3. What cables do you have, and are they all on the same home network?
4. Do you already have the file `Qwen3.8-Flash-Next-Q4.gguf` (177 GB)?

Inspect this computer yourself. Do not ask for anything you can read. Windows is not
supported, and a computer that cannot run Fronk is not a failure: say so and go on.

## 3. Choose how to reach the nodes

Offer these two and recommend the first:

- **SSH from here.** The person enables it once on each node, and you do the rest from
  the head. On a Mac: System Settings, General, Sharing, Remote Login. On Linux, offer
  the exact command. They approve the connection; never ask for passwords or keys.
- **An agent on each computer.** They open Codex or Claude Code on each node. Write them
  one short text to paste there, with that node's name and what to do. They carry the
  pairing code by hand in a terminal prompt, because agents must not see it.

## 4. Connect the computers

Follow `docs/llm/NETWORK.md`. Decide the network yourself and present it as a plan they
can approve. The home network is enough to begin with if all computers are on it. Propose
direct Thunderbolt or Ethernet cables only if they have them, and explain the gain in one
line. You choose the addresses and apply them. They plug in cables and say yes.

## 5. Install

Follow "Drive the installation without a terminal" in `docs/llm/INSTALL.md`: head first,
then each node, then the synthetic demo. The demo needs no model. Show its verdict in plain
words: whether the model fits, and the estimated speed. Stop and ask before downloading
177 GB.

On each Mac node, ask them to log in there and accept **Local Network** when asked.

## 6. Finish

Select or download the exact model, start the head, and wait until it answers. Ask a real
question through the API and show the answer. Then give them:

- The chat address: open `~/fronk/bin/fronk-console` on the head.
- How to start and stop it again.
- What is running on which computer.
- Anything you could not test.

If something fails, follow `docs/llm/DEBUG.md`. Do not bypass pairing, hashes, numerical
tests or the demo before download to make progress.
