"""Export a self-contained public tree without copying a developer's Git history.

The installer allowlist is the base; source-files adds examples and GitHub files.
Refuse existing destinations and symlinks. The result can be tested in isolation
and initialized as a new repository, without exposing local state or old commits.
"""
from pathlib import Path
import gzip
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def selected_files(destination):
    files = set()
    for manifest in ("tools/install/runtime-files", "tools/release/source-files"):
        for line in (ROOT / manifest).read_text().splitlines():
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid allowlist path: {name}")
            path = ROOT / relative
            if not path.exists():
                raise ValueError(f"Missing release input: {name}")
            if path.is_dir() and destination.resolve().is_relative_to(path.resolve()):
                raise ValueError(f"Output directory is inside release input: {name}")
            for entry in [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]:
                rel = entry.relative_to(ROOT)
                if "__pycache__" in rel.parts or entry.name == ".DS_Store":
                    continue
                if entry.is_symlink():
                    raise ValueError(f"Symlink in release input: {rel}")
                if not entry.is_file():
                    continue
                if (entry.suffix in {".gguf", ".pyc", ".log"} or
                        any(p in {".git", ".cluster", ".env", "cluster.secret"} for p in rel.parts)):
                    raise ValueError(f"Local state in release input: {rel}")
                files.add(rel)
    return sorted(files)


def export(destination):
    destination = destination.absolute()
    archive = destination.with_name(destination.name + ".tar.gz")
    checksum = archive.with_name(archive.name + ".sha256")
    if any(p.exists() or p.is_symlink() for p in (destination, archive, checksum)):
        raise ValueError("Destination or archive already exists; choose a new output directory")
    files = selected_files(destination)
    for path in files:
        if (ROOT / path).is_relative_to(destination.resolve()):
            raise ValueError("Output directory contains release inputs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fronk-export-", dir=destination.parent) as temp:
        stage = Path(temp) / destination.name
        stage.mkdir()
        for relative in files:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        subprocess.run([sys.executable, str(stage / "tools/check-docs.py")], check=True)
        subprocess.run([sys.executable, str(stage / "tools/install/package-manifest.py"), str(stage)], check=True)
        # Sorted names, fixed timestamps and no local uid/gid in the source tar.
        packed = Path(temp) / archive.name
        with packed.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(stage.rglob("*")):
                    if not path.is_file():
                        continue
                    info = tar.gettarinfo(str(path), arcname=str(Path(destination.name) / path.relative_to(stage)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
        digest = hashlib.sha256(packed.read_bytes()).hexdigest()
        stage.rename(destination)
        packed.rename(archive)
        checksum.write_text(f"{digest}  {archive.name}\n")
    size = sum((destination / p).stat().st_size for p in files)
    print(f"Exported {len(files)} files, {size:,} source bytes: {destination}")
    print(f"{digest}  {archive}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python3 tools/release/export-source.py NEW_DIRECTORY")
    try:
        export(Path(sys.argv[1]).expanduser())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Source export failed: {error}")
