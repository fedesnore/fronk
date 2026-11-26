# The llama.cpp head and the 3-bit weights

Fronk 0.3.0 runs one file, `Qwen3.8-Flash-Next-Q4.gguf`, with a head built on
DwarfStar/ds4. This checkout adds a second way to run the same model: a 3-bit
conversion of its weights (GSQ-RCO), with llama.cpp as the head's dense engine.
This page explains why the engine had to change, which program does which part
of the work, the two kinds of node, and how to build and start it.

It is a development path. You build it from a checkout, on an Apple Silicon head;
the downloadable 0.3.0 managers do not contain it. The ds4 head and the Q4 file
work as before. [What changed in 0.4.0](releases/0.4.0.md) lists everything that
came with it.

**On this page:**
[why a second engine](#why-a-second-engine) ·
[who computes what](#who-computes-what) ·
[two kinds of node](#two-kinds-of-node) ·
[the unit format](#the-unit-format) ·
[build and start](#build-and-start) ·
[checks](#checks) ·
[limits](#limits)

## Why a second engine

The reason is memory, and then compatibility.

| | Q4 file, ds4 head | GSQ-RCO IQ3_XXS, llama.cpp head |
|---|---:|---:|
| Model on the head's disk | 177.28 GB | 47.04 GB, plus a 28.80 GB n-gram table |
| One expert unit | 3.35 GB | 2.185 GB |
| All 20 units, in the cluster's RAM | 66.94 GB | 43.70 GB |

With the Q4 file the experts take about 67 GB of RAM across the cluster. The
conversion that ISTA-DASLab published brings them under 44 GB, and its authors
report a task average of 92.57 where the unquantized model scores 93.12. On a
desk of ordinary computers, 23 GB less is the difference between filling every
machine and being free to put the work where it runs fastest.

The obstacle was reading the file. GSQ-RCO chooses a quantization type for every
tensor separately. The expert matrices use IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and
IQ3_S on the input side and IQ4_NL or Q2_0 on the output side, a different pair
in each layer; attention and the shared expert add IQ4_XS and three K types. ds4
does not implement IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS or Q2_0, and
since the dense part uses them too, it cannot run this file at all.

llama.cpp 0.9.0 reads all of them. It also knows this architecture, drafts with
MTP and can leave the n-gram table on disk. So Fronk attaches to llama.cpp
instead of teaching ds4 seven formats: the whole integration is one patch,
about 750 added lines against a pinned public commit
(`tools/engine/gsq/llama-fronk.patch`).

The nodes follow the same rule. Their `gsq` backend computes the input rows with
ggml's own kernels for those types (a small part of ggml is vendored in
`third_party/ggml`), or with Fronk's multi-token versions, which are checked bit
for bit against ggml's. Only the output side has kernels of its own, because a
32-neuron slice takes half of a Q2_0 block.

## Who computes what

Three programs run on the head Mac, and one on every node:

```text
chat client ──HTTP──> fronk_engine.py ──HTTP──> llama.cpp engine ──local socket──> fronk head ──UDP──> nodes
```

| Program | Where | Its part |
|---|---|---|
| `fronk-llama-server` (llama.cpp with the patch) | Head, GPU | Everything that is not a routed expert: embeddings, attention and recurrent state, the router that picks ten experts, the shared expert, the logits, MTP drafts and their verification |
| `fronk head` | Head, CPU | The cluster: it measures machines and links, decides who holds which units, prepares and sends them, and at every layer sends the request and adds the partial results |
| `tools/fronk_engine.py` | Head | Supervises the engine: serves the API, queues requests, guards memory, saves the chat when something stops |
| `fronk node` | Each node | Holds units and computes its slices of the ten routed experts of every token. No dense computation and no shared expert: its gate in the request is 0, and the backend refuses any other value |

The patch changes one place in llama.cpp's graph. In each of the 48 trunk layers
the router still runs in the graph. Where the routed experts would be multiplied,
a custom operation hands the layer's input, the ten expert ids and their weights
to `fronk head` through a local socket ([fronk_client.h](../client/fronk_client.h))
and waits for the sum. While the nodes work, the GPU computes the shared expert
of the same layer. The expert matrices are kept off the GPU and are not read
while the cluster answers.

Without `FRONK_MOE_SOCKET` in its environment the patched binary is plain
llama.cpp, so it can be compared with itself. `FRONK_MOE_LOOPBACK=ref` computes
the routed sum locally with ggml's reference dequantizers, to check that the
hook changes nothing (run it with `--no-repack`). `FRONK_MOE_LOOPBACK=zero`
returns zeros, to measure what leaving the GPU 48 times per pass costs.

A few consequences of this division:

- **Prompt batches are wider.** The engine reads a prompt 64 tokens at a time and
  sends each layer as two calls of 32, both in flight at once. The limit of one
  call rose from 8 to 32 tokens for every backend.
- **Requests can travel quantized.** The IQ kernels take their input as Q8_K. The
  head can quantize each activation once and send that form: 2,920 bytes instead
  of 5,120 as FP16, 3,112 per token with the routing record and headers. Replies
  stay FP16. This is opt-in (a file named `wire-q8k` in the head's state
  directory), and the head refuses it unless every node confirms it.
- **Drafts stay on the head.** MTP drafts come from a small separate GGUF, loaded
  as a draft model on the GPU; its experts never go to the cluster. Every drafted
  token is verified by the full model in the same pass, and by default only the
  tokens the full model would have written are kept (`gsq-lenience` 0).
- **A stalled node pauses the chat.** An expert sum has no side effects, so the
  head asks the same call again with growing deadlines, for about 30 s. If the
  cluster does not come back, the engine can finish the pass in flight by itself
  from the model file, which lets the supervisor save the chat before stopping.

## Two kinds of node

Every node described so far is a **slice node**. It holds units, so it owns a
share of every expert and answers every layer call. Slice nodes are the cluster:
together they must cover all 20 units.

The GSQ path adds an optional second kind, the **expert cache**. Routing is
uneven: about a tenth of a layer's experts receive half of the calls. A cache
keeps some of those experts whole, all 20 slices of each, in locked memory, and
computes them alone. For that call the slice nodes are asked only for the
experts that no cache holds, and the partial results add up to the same sum.

| | Slice node | Expert cache |
|---|---|---|
| Holds | Units: the same 32 neurons of every expert | Whole experts, the ones the chat is using now |
| Required | Yes | No: an expert it does not hold is computed by the slices |
| Which machine | Any node, or the head | The head itself, or a node that holds no unit and has the model file on its own disk |
| Contents change | Only when the plan changes | Continuously: the head orders each copy and each release, a few per token |
| If it stops answering | The call is asked again; the chat waits | Its experts go back to the slices |

A cache trades memory for latency on machines that are fast but hold little, and
it never becomes a dependency: an expert missing from every cache is neither an
error nor a disk read on the request path.

Caches are off unless the head's state directory has a file named `cache-node`,
with one line per cache: `NAME GB [SHARE]`, at most three. `NAME` is the head's
own name or a linked node without units. Without the file every expert is
computed by slices. [hot.h](../src/hot.h) describes the store, the policy that
follows the chat and the files it keeps.

## The unit format

A unit is still the same 32 neurons of every routed expert in every layer, cut
from the file without changing a weight.

- Gate and up rows span the 2,560 inputs and are copied whole.
- Down rows span the 640 neurons. IQ4_NL blocks are 32 weights, so a slice takes
  one block (18 bytes). Q2_0 blocks are 64 weights, so a slice takes the block's
  scale and its half of the 2-bit codes (10 bytes).
- The shared expert is not in the unit. The engine computes it.

The types of each layer are a fixed table for the pinned file; `fr_gsq_open`
refuses any other file. A head takes the format from its model file and tells
every node at connection (`LAYOUT gsq`). A node computes one format at a time and
keeps GSQ units in `state/gsq`, apart from native ones, so switching never
overwrites verified files. Timing profiles are stored per format.
[layout.h](../src/layout.h) and [gsq.h](../src/gsq.h) are the references.

## Build and start

You need an Apple Silicon Mac for the head, with the Xcode command-line tools,
CMake and network access for the first build. Nodes need what they needed before.

1. **Build the engine** on the head, in the checkout:

   ```sh
   make gsq-engine
   ```

   This fetches llama.cpp at commit `cd959bae` (0.9.0), applies the patch and
   builds `build/gsq-engine/fronk-llama-server` and a small GPU keepalive.

2. **Get the weights.** Put the two IQ3_XXS shards of the pinned revision in one
   folder; [MODEL.md](MODEL.md#the-3-bit-weights-for-the-llamacpp-head) gives
   names, sizes and digests.

3. **Build the MTP file** next to that folder (optional, but without it the
   engine drafts from n-grams only, which is slower):

   ```sh
   python3 tools/gsq-mtp-build.py FOLDER/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
       FOLDER/../mtp-Qwen3.8-Flash-Next.gguf
   ```

   It takes the MTP layer (1.51 GB) from the original Q4 file, by HTTP ranges or
   from a local copy with `--source`, and the embeddings from the GSQ shard.
   Nothing is requantized.

4. **Install the head** from the checkout with the engine chosen once:

   ```sh
   bash tools/install/fronk-install.sh --role head --engine gsq
   ```

   The choice is kept in the installed configuration. The rest of the workflow
   is the one in the [installation guide](INSTALL.md): nodes, synthetic demo,
   model selection, plan. Install the nodes from the same source, because the
   head and the nodes must agree on the transport.

Engine settings are one-line files in the head's state directory; remove a file
to return to its default. The header of
[run.sh](../tools/engine/gsq/run.sh) lists all of them. The ones used most:

| File | Meaning | Default |
|---|---|---|
| `ds4-ctx` | Context length in tokens | 65536 |
| `gsq-draft` | MTP draft tokens per pass, 1–15 | 7 |
| `gsq-draft-p-min` | Stop a draft below this probability | 0.5 |
| `gsq-lenience` | Accept drafted tokens the full model would not have written | 0 (exact) |
| `gsq-thinking` | Present: the model reasons before answering | absent |
| `gsq-no-vision` | Present: the image encoder is not loaded | absent |

## Checks

| Check | What it covers |
|---|---|
| `make DS4= test-gsq` | Unit sizes and names, the `gsq` backend's self-test against the scalar oracle, several tokens sharing experts |
| `make DS4= test-q8-wire` | Encoding and reassembly of Q8_K requests |
| `tools/gsq-multicheck.c` | The multi-token kernels against ggml's single-token ones, bit for bit |
| Head start | Before serving, the head sends one layer through the real cluster and compares the sum with an oracle computed from the model file, for a 2-token call and for a 32-token one |
| `FRONK_MOE_LOOPBACK=ref` | The patched engine against the unpatched graph |

## Limits

- **One exact file.** Revision `ed59f92`, IQ3_XXS. Another quantization of the
  same model has another type table and is refused.
- **Apple Silicon head only.** The installer refuses `--engine gsq` elsewhere.
- **CPU nodes only.** The `gsq` backend runs on x86 with AVX2, FMA, F16C and
  BMI2, or on ARM with NEON. There is no Metal backend for GSQ units, and RAM +
  SSD placement reads native units only.
- **Not a qualified release.** This path has run on the author's test cluster;
  it has not been through the packaged-installer qualification in
  [RELEASE.md](RELEASE.md), and no manager contains it.

[PLAN-gsq-rco.md](PLAN-gsq-rco.md) keeps the working notes of the port, with the
measurements taken along the way.
