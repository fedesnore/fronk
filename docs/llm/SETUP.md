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

## 2. Ask how many nodes and of what type

Inspect this computer yourself first. Then ask only this, in one message:

1. How many node computers are you connecting? This one, the head, is not counted.
2. What type is each one: Mac or Linux, and roughly which model and how much memory? A
   rough description is enough, and you will read the details over SSH.
3. Is SSH already enabled on them? If they do not know, the answer is "not yet".
4. What cables do you have, and are they all on the same home network?
5. Do you already have the file `Qwen3.8-Flash-Next-Q4.gguf` (177 GB)?

Suggest a Mac with Apple Silicon as the head; other heads are experimental. Windows is not
supported: a computer that cannot run Fronk is not a failure, so say so and go on.

## 3. Open SSH from the head to each node

You run on the head and do everything on the nodes over SSH: that is how you send them the
software and check their connections. Set it up node by node, over the network the
computers already share, before anything else.

**If SSH is not enabled on a node, guide them there, one step at a time:**

- Mac: System Settings, General, Sharing, turn on **Remote Login**.
- Linux: in a terminal on that computer, `sudo apt install openssh-server` and
  `sudo systemctl enable --now ssh`. Adapt to its distribution. The `sudo` password is
  typed there, by them.
- Then ask them to read you two things from a terminal on that node, its user name and
  address: `whoami`, and on a Mac `ipconfig getifaddr en0` or on Linux `hostname -I`.
  Try `NAME.local` or `dns-sd -B _ssh._tcp` first, so they read less.

**Then connect, with the password typed in their terminal, never to you:**

1. Create a key just for this setup, if there is none:
   `ssh-keygen -t ed25519 -N '' -f ~/.ssh/fronk_setup`. Say what it is and that they can
   remove it at the end.
2. Give them one command per node to run in their own terminal on the head:
   `ssh-copy-id -i ~/.ssh/fronk_setup.pub USER@HOST`. Tell them it may ask to accept the
   computer, then for that computer's password, and that nothing appears while they type.
   You never see the password, and you never ask for it in this conversation.
3. Check each one: `ssh -i ~/.ssh/fronk_setup -o BatchMode=yes -o IdentitiesOnly=yes USER@HOST 'uname -sm; hostname'`.
   Use this key for every command below. Compare what you read with what they told you.

If SSH reports `Host key verification failed`, do not turn the check off or delete the old
key yourself. The computer is new to this one, or it changed: ask them to confirm it is theirs,
then to connect once in their own terminal and accept it.

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
