"""Inventory the allowlisted payload; never traverse symlinks outside it."""
from pathlib import Path
import hashlib
import sys

root = Path(sys.argv[1]).resolve(strict=True)
lines = []
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"Package contains a symlink: {path.relative_to(root)}")
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    if relative == "SOURCE-SHA256SUMS":
        continue
    if "\n" in relative or "\r" in relative:
        raise SystemExit("Invalid package filename")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(data)
    lines.append(f"{checksum.hexdigest()}  {relative}\n")
(root / "SOURCE-SHA256SUMS").write_text("".join(lines))
