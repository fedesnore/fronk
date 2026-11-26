#!/usr/bin/env python3
"""gsq-mtp-build.py GSQ_SHARD1 OUT [--source FILE_OR_URL]

Build an MTP-only GGUF for llama.cpp's draft-mtp from:
  - the original Qwen3.8-Flash-Next Q4 GGUF (antirez, revision d600fe1): its
    metadata (49 blocks, nextn_predict_layers 1, tokenizer) and the 32 tensors of
    its MTP layer blk.48 (1.51 GB), read from a local copy or by HTTP ranges;
  - the GSQ-RCO shard: token_embd and output (the draft's embedding and logits).
llama.cpp treats a file without trunk layer 0 as MTP-only. Nothing is requantized.
"""
import argparse, struct, sys, urllib.request

URL = ("https://huggingface.co/antirez/qwen3.8-flash-next-gguf/resolve/"
       "d600fe1a43d2e1cdcadb85144ce3142f66f9eefe/Qwen3.8-Flash-Next-Q4.gguf")
SIZES = {0: (4, 1), 1: (2, 1), 30: (2, 1), 8: (34, 32), 12: (144, 256), 13: (176, 256),
         14: (210, 256), 21: (110, 256), 39: (17, 32)}

class Source:
    def __init__(self, where):
        self.url = where if where.startswith("http") else None
        self.f = None if self.url else open(where, "rb")
    def read(self, off, n):
        if self.f:
            self.f.seek(off); return self.f.read(n)
        out = bytearray()
        while len(out) < n:  # 16 MB ranges, each retried: one stalled read must not lose the build
            k = min(16 << 20, n - len(out))
            for attempt in range(8):
                try:
                    out += self._range(off + len(out), k); break
                except (OSError, IOError) as e:
                    print(f"range at {off + len(out)}: {e}, retry {attempt + 1}", file=sys.stderr, flush=True)
                    import time; time.sleep(2 + 3 * attempt)
            else:
                raise IOError(f"source unreachable at {off + len(out)}")
        return bytes(out)
    def _range(self, off, n):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={off}-{off + n - 1}"})
        out = bytearray()
        with urllib.request.urlopen(req, timeout=60) as r:
            while chunk := r.read(1 << 22):
                out += chunk
        if len(out) != n: raise IOError(f"short read at {off}: {len(out)} of {n}")
        return bytes(out)

def parse(head):
    p = 4
    def rd(f, n):
        nonlocal p; v = struct.unpack_from(f, head, p)[0]; p += n; return v
    def s():
        nonlocal p; n = rd("<Q", 8); v = head[p:p + n].decode(); p += n; return v
    fm = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et = rd("<I", 4); n = rd("<Q", 8); return [val(et) for _ in range(n)]
        return rd(fm[t], sz[t])
    rd("<I", 4); nt = rd("<Q", 8); nk = rd("<Q", 8)
    kv0 = p; kv = {}
    for _ in range(nk):
        k = s(); kv[k] = val(rd("<I", 4))
    kv_bytes = head[kv0:p]
    # raw bytes of every key/value entry, to recombine metadata without re-encoding
    entries, q = {}, kv0
    while q < p:
        start = q; p_save = p; p = q
        k = s(); val(rd("<I", 4)); entries[k] = head[start:p]; q = p; p = p_save
    tensors = []
    for _ in range(nt):
        n = s(); d = rd("<I", 4); ne = [rd("<Q", 8) for _ in range(d)]; t = rd("<I", 4); off = rd("<Q", 8)
        tensors.append((n, ne, t, off))
    align = kv.get("general.alignment", 32)
    data = (p + align - 1) // align * align
    return nk, kv_bytes, tensors, data, align, kv, entries

def nbytes(ne, t):
    b, w = SIZES[t]; n = 1
    for e in ne: n *= e
    return n * b // w

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gsq"); ap.add_argument("out"); ap.add_argument("--source", default=URL)
    a = ap.parse_args()
    src = Source(a.source)
    head = src.read(0, 32 << 20)
    nk, kv_bytes, tensors, data, align, kv, src_entries = parse(head)
    if kv.get("general.architecture") != "qwen4exp" or kv.get("qwen4exp.nextn_predict_layers") != 1:
        sys.exit("source is not the Qwen3.8-Flash-Next file with one MTP layer")
    mtp = [t for t in tensors if t[0].startswith("blk.48.")]
    with open(a.gsq, "rb") as g:
        ghead = g.read(32 << 20)
    _, _, gt, gdata, _, _, g_entries = parse(ghead)
    # Metadata: the GSQ file's (rope sections, tokenizer and template as the trunk
    # that runs), without its split keys, plus the source's MTP shape: 49 blocks,
    # one nextn layer, 49 compress ratios.
    # The PLE table (per-layer n-gram embedding, shard 2) feeds trunk layer 1 only,
    # so the MTP-only file drops its keys and needs no per_layer_token_embd.
    meta = {k: v for k, v in g_entries.items() if not k.startswith(("split.", "qwen4exp.ple."))}
    # encoded with the GSQ file's types (llama.cpp checks array element types)
    def entry(k, t, v, et=None):
        b = struct.pack("<Q", len(k)) + k.encode() + struct.pack("<I", t)
        if t == 9: return b + struct.pack("<IQ", et, len(v)) + struct.pack(f"<{len(v)}i", *v)
        return b + struct.pack({4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t], v)
    def type_of(k):
        e = g_entries[k]; return struct.unpack_from("<I", e, 8 + len(k))[0]
    meta["qwen4exp.block_count"] = entry("qwen4exp.block_count", type_of("qwen4exp.block_count"), kv["qwen4exp.block_count"])
    meta["qwen4exp.nextn_predict_layers"] = entry("qwen4exp.nextn_predict_layers", type_of("qwen4exp.block_count"), kv["qwen4exp.nextn_predict_layers"])
    meta["qwen4exp.attention.compress_ratios"] = entry("qwen4exp.attention.compress_ratios", 9, kv["qwen4exp.attention.compress_ratios"], 5)
    if "general.alignment" not in meta: meta["general.alignment"] = src_entries["general.alignment"]
    nk, kv_bytes = len(meta), b"".join(meta.values())
    shared = [t for t in gt if t[0] in ("token_embd.weight", "output.weight")]
    if len(mtp) != 32 or len(shared) != 2: sys.exit(f"found {len(mtp)} MTP and {len(shared)} shared tensors")
    plan = [(n, ne, t, "src", data + off) for n, ne, t, off in mtp] + [(n, ne, t, "gsq", gdata + off) for n, ne, t, off in shared]
    info, off = b"", 0
    for n, ne, t, _, _ in plan:
        info += struct.pack("<Q", len(n)) + n.encode() + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        info += struct.pack("<IQ", t, off)
        off = (off + nbytes(ne, t) + align - 1) // align * align
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(plan), nk) + kv_bytes + info
    pad = (len(hdr) + align - 1) // align * align - len(hdr)
    total = 0
    import os
    tmp, start = a.out + ".tmp", len(hdr) + pad
    have = 0
    if os.path.exists(tmp):
        with open(tmp, "rb") as o:
            if o.read(start) == hdr + b"\0" * pad: have = os.path.getsize(tmp)
    with open(tmp, "r+b" if have else "wb") as o, open(a.gsq, "rb") as g:
        if not have: o.write(hdr + b"\0" * pad)
        pos = start
        for n, ne, t, where, at in plan:
            size = nbytes(ne, t)
            padded = (size + align - 1) // align * align
            if pos + padded <= have:  # already written by an earlier run
                pos += padded; total += size
                print(f"{n:40s} {size / 1e6:8.1f} MB kept", flush=True); continue
            o.seek(pos); o.truncate(); pos += padded
            if where == "gsq":
                g.seek(at); blob = g.read(size)
            else:
                blob = b"".join(src.read(at + k, min(256 << 20, size - k)) for k in range(0, size, 256 << 20))
            o.write(blob); o.write(b"\0" * ((size + align - 1) // align * align - size))
            total += size
            print(f"{n:40s} {size / 1e6:8.1f} MB from {where}", flush=True)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {len(plan)} tensors, {total / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
