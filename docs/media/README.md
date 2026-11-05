# Fronk — It could work

## The idea

One object, one question: **where do you cut it?** The model is a glass block with its three
true axes: 640 neurons, 512 experts, 48 layers. For the first 88 seconds nothing cuts away from
it, on black. The existing architectures for distributing a model are literally planes through that
block: by layer (pipeline), by expert, every matrix (tensor, the dense part included), each
named as such with its problem on mixed machines at home, then the option of no cut at all. Then the model's own seam (dense core and
experts) explains why this model, and Fronk's cut lets the light in: the orange slice widens
and wipes the film to white. From there each drawing lands on the real capture at the same
geometry, because the point has become "this ran". It ends where the project's own joke does:
Fronkhensteen off the laboratory table, first stitches, "It could work!".

It is written to work muted, with no voice-over. Type sits on one left margin, measurements are
in mono and attached to the object they measure, and the machines are to-scale elevations
(MacBook Pro 31 cm wide, iMac 65 × 51, Shuttle 20 × 4.3).

| Time | Scene | What it has to do |
|---|---|---|
| 00:00 | question | Silence, then the four machines on the first four beats |
| 00:05 | model | Qwen3.8-Flash-Next: capable, open, a mixture of experts |
| 00:10 | no-fit | To scale: 67 GB does not fit any machine; 104 GB exists across four |
| 00:17 | where | The block's dimensions; says the next scenes are today's architectures |
| 00:22 | pipeline-parallelism | Existing architecture 1 of 3, with its problem at home |
| 00:28 | expert-parallelism | Existing architecture 2 of 3, with its problem at home |
| 00:36 | tensor-parallelism | Existing architecture 3 of 3; it cuts the dense part too |
| 00:44 | no-distribution-ssd | The alternative of not distributing at all |
| 00:48 | your-desk | They were built for racks |
| 00:51 | the-seam | MoE as the enabler: dense head built with DwarfStar, commodity nodes, 10 of 512 |
| 01:10 | fronks-cut | Twenty slices; why there is no talk; the light |
| 01:28 | share-real-plan | The real allocation, landing on the console's strip |
| 01:34 | one-layer-real-report | One layer in slow motion at measured times, landing on the report; cables |
| 01:46 | install | Head first, each node, demo before download, weights, plan |
| 02:00 | chat-and-result | Real chat at real speed; the separate 15.61 test |
| 02:10 | stuck-llm-briefs | The help already in the box, and how to use it |
| 02:24 | first-version-contribute | First version; where hands are wanted; share your cluster |
| 02:41 | end | The wordmark, cut and stitched; credits |

Scene length follows reading time, with a brisker opening up to the existing architectures: the author asked for a film that can really be read, up to
three minutes. A new line or scene gets its words' time before anything else moves.

## Where the numbers come from

Dimensions, unit size and the slice formula are in `PROJECT.md` and `src/model.h` of the repository. 66.94 GB is
20 × 3,347,005,440 bytes.

**≈1.3 GB per token** is arithmetic, not a measurement: 2,713,600 bytes for one routed expert in
one layer × 10 experts × 48 layers. Repeated experts and cached pages can reduce real reads.

**≈5.4 GB dense core** is the dense and output weights figure in `PROJECT.md` for the Vulkan
head; context and scratch space come on top. The drawing's proportion is schematic.

**Installed RAM to scale:** 24 GiB, 48 GiB, 16 GiB, about 16 GB: 104 in round numbers, mixing
GiB and GB. It is installed memory; systems and the dense model need their share.

**The plan and report** are the packaged run of 2026-11-05 in `docs/TEST_CLUSTER.md`: head 1 RAM
+ 1 SSD, iMac 11, M1 Pro 3, Shuttle 4. Unit numbers and colours are read from the console
capture, and the animated strip and bars use the capture's geometry. The previous cut paired the
15.61 result with the earlier 2 + 1 plan; this one does not. In the slow-motion layer, compute
and arrival are the report's figures (M1 Pro 0.73/1.31, iMac 1.82/2.28, Shuttle 0.81/2.04, head
1.02/1.03 ms). How the remainder splits between outbound and return is a drawing: half each.

**15.61 tok/s** is the console's warm short test, labelled as such. The chat excerpt is a separate
request at about 11–12 tok/s, shown with original timestamps; its initial wait and beginning are
trimmed.

**The drawings of the failed approaches are explanations, not benchmarks.** The grid of 512 × 48
with exactly ten lit in each row uses random positions; real routing is not uniform.
The load bars in the expert cut are illustrative too. All the program pictures are captures;
setup is condensed. The box labelled "your frontier model" deliberately draws no interface.

## Credits

The tested head's dense engine is DwarfStar/ds4 by antirez (Salvatore Sanfilippo), and the
experimental Vulkan adapter builds on llama.cpp; see `THIRD_PARTY.md` in the repository.

## Music credit — keep with reposts

“Surf Shimmy” — Kevin MacLeod ([incompetech.com](https://incompetech.com/)).
Licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
[Original track](https://incompetech.com/music/royalty-free/index.html?Search=Search&isrc=USUAN1700018),
ISRC USUAN1700018. Delayed by 1.919 s so its bars fall on the scene changes, with a twelve-bar and a
four-bar section repeated to fit the film, level changes, an end fade and sound effects added. The film opens on three seconds of silence.

The music is **not MIT licensed**. Its attribution is in the closing frame and movie metadata;
retain this credit and the source and license links in a public post's description. The sound
effects are synthesized noise and clicks, with no melody, voice or sample.

Download provenance is recorded in `music-provenance.json`. That record is for verification;
the standalone music download is not shipped in this repository.

## Posting

This is the landscape cut used on GitHub and the project site. Link the
project's actual repository, and describe the rate as a warm short test on this cluster, not a
promise for every model or conversation.
