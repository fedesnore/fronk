# Supported model and weights

Use **Qwen3.8-Flash-Next-Q4.gguf** on the head. After the synthetic demo, choose
**download** in the manager, or **local** if you already have it. The default
download folder is `~/fronk/models`; you can choose another folder or an external
SSD. Keep that drive attached while using Fronk.

Only the head needs the complete file. Fronk verifies it and extracts compact
neuron units for the nodes automatically. You do not need to split it yourself
or copy the full GGUF to each machine. This release requires the exact conversion
identified below; other quantizations need matching implementation support.

| Field | Value |
|---|---|
| File | `Qwen3.8-Flash-Next-Q4.gguf` |
| Repository | `antirez/qwen3.8-flash-next-gguf` on Hugging Face |
| Pinned revision | `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe` |
| Exact bytes | `177280286720` (177.28 GB / 165.11 GiB) |
| SHA-256 | `680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a` |
| Fronk unit bytes | `3347005440`, 20 units total |

[Download the pinned file](https://huggingface.co/antirez/qwen3.8-flash-next-gguf/resolve/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe/Qwen3.8-Flash-Next-Q4.gguf).
The installer offers this download after the synthetic cluster demo and resumes
an incomplete download in the selected folder. Prefer using the manager so it
also verifies the digest and prepares the per-unit manifest.

The package contains main/MTP weights and a BF16 n-gram table read from SSD;
that table is not made fully resident. See the pinned
[upstream model card](https://huggingface.co/antirez/qwen3.8-flash-next-gguf/blob/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe/README.md).
The complete file is needed on the head, even though expert units transferred
to nodes total about 66.94 GB. Keep the original GGUF on a connected SSD.

Weights remain under the upstream
[Qwen Community License](https://huggingface.co/antirez/qwen3.8-flash-next-gguf/blob/d600fe1a43d2e1cdcadb85144ce3142f66f9eefe/LICENSE),
separately from Fronk's MIT code license. Fronk release archives contain no model
weights. Model authors/converters are credited in the upstream model card.

## The 3-bit weights for the llama.cpp head

A head installed with `--engine gsq` uses a different conversion of the same
model, ISTA-DASLab's GSQ-RCO in its IQ3_XXS variant. [GSQ.md](GSQ.md) explains
why it needs another engine and how to build it. The head accepts exactly these
files, two shards kept in one folder:

| Field | Shard 1 (loaded) | Shard 2 (n-gram table, read from SSD) |
|---|---|---|
| File | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` |
| Exact bytes | `47039860096` (47.04 GB) | `28800138432` (28.80 GB) |
| SHA-256 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Both come from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` on Hugging Face,
[revision `ed59f92`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS),
under the license stated in that repository. A GSQ unit is about 2.185 GB, and
the 20 units total 43.70 GB. `tools/install/model-gsq.conf` is the installer
identity for these files.

MTP drafting needs one more file, `mtp-Qwen3.8-Flash-Next.gguf`, in the folder
above the shards. `tools/gsq-mtp-build.py` builds it from the MTP layer of the Q4
file in the table at the top of this page and the embeddings of shard 1, without
requantizing anything. Without it the engine drafts from n-grams only. The
result holds weights from both files, so both licenses apply to it.

## Local files and external drives

Select the exact file or its containing folder in the manager. It checks the
file in place, including paths containing spaces; there is no extra 177 GB copy.
Only the head needs it. Each node receives its own verified native units.

The first SHA-256 scan is intentionally complete. A cache records path, device,
inode, size and nanosecond modification/change times. Changing or replacing the
file triggers verification again. The cache is a local optimization, not a way
to accept a different model. Allow removable-volume access to the application
actually running the installer on macOS.

A frontier model can help with a model, slice or transfer problem: give it
[`docs/llm/MODEL.md`](llm/MODEL.md).

## Changing model support

`tools/install/model.conf` is the installer identity; `model-units.sha256`
contains the expected native unit digests. `src/model.h` and `native.h` define
dimensions and quantized layout; `prep.c` extracts that layout. The dense engine
adapter must agree with all of them.

A model-support change needs a new pinned source/digest, extraction and unit
manifest, independent reference checks for every backend, dense-state/logit
checks, and a real full-cluster generation test. Never reuse this manifest or
silently requantize weights for a different file.

## If selection, verification or transfer stops

Keep the file and the error message. A resumed download should use the same
folder. For external-drive access or a failed unit transfer, follow
[troubleshooting](TROUBLESHOOTING.md), or give an assistant the
[model debugging brief](llm/MODEL.md). It follows the checks from file identity
to the received slice without asking you to start the download again blindly.
