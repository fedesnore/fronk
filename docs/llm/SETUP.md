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
line=$(LC_ALL=C awk '/^__FRONK_PAYLOAD__$/ {print NR + 1; exit}' fronk-manager.sh)
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

## 3. Open SSH from the head to each node

You work from the head: SSH is how you send each node its software and check its
connections. Set it up first, over the network the computers already share.

1. Ask them to enable it once on each node. On a Mac: System Settings, General, Sharing,
   Remote Login. On Linux, give the exact command for that computer.
2. Find each node yourself: try `NAME.local`, `dns-sd -B _ssh._tcp` on a Mac, or the
   addresses on this network. Ask them only for what you cannot find, such as the user
   name on that computer.
3. Create a key just for this setup, if there is none:
   `ssh-keygen -t ed25519 -N '' -f ~/.ssh/fronk_setup`. Say what it is and that they can
   remove it at the end.
4. Ask them to run one command per node in their own terminal, on the head, and type that
   computer's password there: `ssh-copy-id -i ~/.ssh/fronk_setup.pub USER@HOST`. You never
   see the password, and you never ask for it.
5. Check each one: `ssh -i ~/.ssh/fronk_setup -o BatchMode=yes USER@HOST 'uname -sm; hostname'`.
   Use this key for every command below.

If a node cannot be reached, say which computer and what to check there, one thing at a
time. Do not go on until every node answers. If an action needs an administrator password
on a node, give them the exact command to run in a terminal on that computer.

## 4. Connect the computers

Follow `docs/llm/NETWORK.md`. Inspect every computer over SSH and decide the network
yourself, then present it as a plan they can approve. The home network is enough to begin
with if all computers are on it. Propose direct Thunderbolt or Ethernet cables only if they
have them, and explain the gain in one line. You choose the addresses and apply them. They
plug in cables and say yes.

Keep the route you are using for SSH while you change a link, so you can undo the change.
Then check every link from both ends: the node's address, `ping`, and the route the head
uses to reach it.

## 5. Install

Follow "Drive the installation without a terminal" in `docs/llm/INSTALL.md`: head first,
then each node, then the synthetic demo. The demo needs no model. Show its verdict in plain
words: whether the model fits, and the estimated speed. Stop and ask before downloading
177 GB.

On each Mac node, ask them to log in there and accept **Local Network** when asked. You
cannot accept it over SSH.

## 6. Finish

Select or download the exact model, start the head, and wait until it answers. Ask a real
question through the API and show the answer. Then give them:

- The chat address: open `~/fronk/bin/fronk-console` on the head.
- How to start and stop it again.
- What is running on which computer.
- Anything you could not test.

Fronk does not use SSH while it runs. Offer to remove the setup key, from each node's
`~/.ssh/authorized_keys` and from the head, and tell them they can turn Remote Login off.

If something fails, follow `docs/llm/DEBUG.md`. Do not bypass pairing, hashes, numerical
tests or the demo before download to make progress.
