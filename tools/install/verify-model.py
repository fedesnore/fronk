"""Verify the pinned GGUF once, then seed Fronk's regular per-unit manifest.

model-units.sha256 is the output of fr_manifest for that exact GGUF. The native
head still hashes units during transfer, and still builds its own manifest for
models configured outside the installer.

    verify-model.py FILE STATE BYTES SHA256 [--cache NAME --no-manifest]

--cache names the verification stamp (one per file when a model has several
shards); --no-manifest only verifies: the GSQ head builds its unit manifest itself.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def identity(path, stat, digest):
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, device=stat.st_dev, inode=stat.st_ino, sha256=digest)


def replace(path, data):
    temp = path.with_name(path.name+".new")
    temp.write_text(data)
    temp.replace(path)


def verify(path, state, size, digest, cache_name="model.verified", manifest=True):
    path = path.resolve(strict=True)
    stamp = identity(path, path.stat(), digest)
    if stamp["bytes"] != size:
        raise ValueError(f"wrong model size ({stamp['bytes']} bytes, expected {size})")
    cache = state / cache_name
    try:
        cached = json.loads(cache.read_text())
    except (FileNotFoundError, ValueError):
        cached = None
    if cached != stamp:
        checksum = hashlib.sha256()
        start = last = time.monotonic()
        read = 0
        print("Checking the existing GGUF in place. Keep the drive connected.", flush=True)
        with path.open("rb") as f:
            while True:
                data = f.read(8*1024*1024)
                if not data:
                    break
                checksum.update(data)
                read += len(data)
                now = time.monotonic()
                if now-last >= 15:
                    rate = read/(now-start)
                    print(f"Verified {100*read/size:.0f}% · {rate/1e6:.0f} MB/s · about {(size-read)/rate/60:.1f} min left", flush=True)
                    last = now
        if checksum.hexdigest() != digest:
            raise ValueError("model SHA-256 does not match the supported GGUF")
        if identity(path, path.stat(), digest) != stamp:
            raise ValueError("the GGUF changed during verification; retry")
        replace(cache, json.dumps(stamp)+"\n")
    if not manifest:
        return
    template = Path(__file__).with_name("model-units.sha256").read_text().splitlines()
    if (len(template) != 21 or template[0] != "gguf-sha256 "+digest or
            not all(re.fullmatch(f"unit-{i:02d} [a-f0-9]{{64}}", line) for i, line in enumerate(template[1:]))):
        raise ValueError("package unit manifest does not match the verified model")
    replace(state/"manifest.txt", f"gguf {size} {stamp['mtime_ns']}\n"+"\n".join(template[1:])+"\n")


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        cache, manifest = "model.verified", True
        if "--no-manifest" in args:
            args.remove("--no-manifest"); manifest = False
        if "--cache" in args:
            i = args.index("--cache"); cache = args[i+1]; del args[i:i+2]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", cache):
            raise ValueError("invalid cache name")
        verify(Path(args[0]), Path(args[1]), int(args[2]), args[3], cache, manifest)
    except (OSError, ValueError) as error:
        sys.exit("Fronk model verification: "+str(error))
