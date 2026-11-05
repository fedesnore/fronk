# Third-party code and model notices

Fronk's own code is MIT licensed; see `LICENSE`. Imported components keep their
own notices. Source and installer distributions must carry these notices too.

| Component | Use | Source / notice |
|---|---|---|
| DwarfStar / ds4 | Apple Silicon dense graph, tokenizer, MTP and model server | https://github.com/antirez/ds4 ; MIT notice in `tools/engine/LICENSE-ds4`, also preserved in staged `ds4/LICENSE` |
| ggml contributors / DeepSeek | Work credited by ds4 | Copyright notices retained in ds4's license/source |
| Iris | ds4 image support | `ds4/third_party/iris/LICENSE` in the staged engine |
| llama.cpp / ggml | Vulkan dense head and GPU probe | https://github.com/ggml-org/llama.cpp ; `LICENSE-llama.cpp` and `licenses/` in each bundled GPU runtime |
| MoltenVK | Vulkan on Intel macOS | https://github.com/KhronosGroup/MoltenVK ; `LICENSE-MoltenVK` beside the bundled library |
| Deep Chat | Browser chat component | `tools/vendor/deep-chat-2.5.1/LICENSE` |
| Qwen model and GGUF conversion | Separately downloaded weights | `docs/MODEL.md` and the pinned upstream model card/license |
| Kevin MacLeod — Surf Shimmy | Instrumental in the demo film | CC BY 4.0; attribution, original track and edit notes in `docs/media/README.md` |

`tools/engine/prepare.sh` pins the public ds4 base commit;
`tools/engine/ds4-fronk.patch` records the Fronk integration and precursor changes.
`tools/mtp/` applies the subsequent worker/MTP patches in a private staging tree.
`tools/gpu-head/build.sh` pins the llama.cpp revision and keeps its notices;
`macos-dist.sh` verifies the MoltenVK archive before extraction.

The preserved earlier Shuttle kernel sources under `port_sources/` document the
AVX2 port's provenance. They are historical reference material, not a runtime
dependency. Model weights, credentials and local machine state are not shipped.
