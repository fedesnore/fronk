"""Verify a release as a set, without executing or extracting its installers.

Archive checksums alone cannot catch accidentally mixing two valid releases.
Compare the source download with the reviewed tree, then require every shared
source file in the installer to match. The embedded engine/runtime has its own
payload manifest; its correctness is qualified by builds and hardware tests.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def manifest(text):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match or match[2] in result:
            raise ValueError('Invalid or duplicate checksum entry')
        name = PurePosixPath(match[2])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe checksum path')
        result[str(name)] = match[1]
    return result


def archive(data, top):
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        for entry in tar.getmembers():
            path = PurePosixPath(entry.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != top:
                raise ValueError('Unexpected archive path: '+entry.name)
            if entry.isdir(): continue
            if not entry.isfile():
                raise ValueError('Non-regular archive entry: '+entry.name)
            relative = str(path.relative_to(top))
            if relative in files: raise ValueError('Duplicate archive entry: '+entry.name)
            files[relative] = tar.extractfile(entry).read()
    return files


def check_inventory(files):
    expected = manifest(files['SOURCE-SHA256SUMS'].decode())
    actual = {name: sha(data) for name, data in files.items() if name != 'SOURCE-SHA256SUMS'}
    if expected != actual:
        differences = sorted(name for name in expected.keys() | actual.keys()
                             if expected.get(name) != actual.get(name))
        raise ValueError('Source inventory differs: '+', '.join(differences))
    return actual


def verify(downloads, source):
    version = re.search(r'^VERSION := (.+)$', (source/'Makefile').read_text(), re.M)[1]
    names = {'fronk-manager.sh', 'fronk-manager-macos.zip', 'fronk-manager-linux.tar.gz',
             f'fronk-source-{version}.tar.gz'}
    checksums = manifest((downloads/'SHA256SUMS').read_text())
    if checksums.keys() != names: raise ValueError('The four release downloads must appear in SHA256SUMS')
    artifacts = {}
    for name in sorted(names):
        data = (downloads/name).read_bytes()
        if sha(data) != checksums[name]: raise ValueError('Download checksum mismatch: '+name)
        artifacts[name] = {'sha256': checksums[name], 'bytes': len(data)}
    shell = (downloads/'fronk-manager.sh').read_bytes()
    with zipfile.ZipFile(downloads/'fronk-manager-macos.zip') as zipped:
        if zipped.read('Fronk Installer Manager.app/Contents/Resources/installer.sh') != shell:
            raise ValueError('macOS app contains a different manager')
    linux = archive((downloads/'fronk-manager-linux.tar.gz').read_bytes(), 'Fronk Installer Manager')
    if linux['installer.sh'] != shell: raise ValueError('Linux launcher contains a different manager')
    marker = b'\n__FRONK_PAYLOAD__\n'
    if marker not in shell: raise ValueError('Installer payload marker missing')
    payload = archive(shell.split(marker, 1)[1], 'fronk')
    payload_hashes = check_inventory(payload)
    public = archive((downloads/f'fronk-source-{version}.tar.gz').read_bytes(), f'fronk-{version}')
    public_hashes = check_inventory(public)
    reviewed = manifest((source/'SOURCE-SHA256SUMS').read_text())
    if public_hashes != reviewed: raise ValueError('Source download differs from the reviewed export')
    for name, digest in reviewed.items():
        if sha((source/name).read_bytes()) != digest:
            raise ValueError('Reviewed file changed after export: '+name)
    # Every runtime input must be present, including guides and helper scripts.
    required = set()
    for name in (source/'tools/install/runtime-files').read_text().splitlines():
        name = name.strip()
        if not name or name.startswith('#'): continue
        required.update(p for p in reviewed if p == name or p.startswith(name+'/'))
    for name in required:
        if payload_hashes.get(name) != reviewed[name]:
            raise ValueError('Installer differs from source: '+name)
    return dict(version=version, artifacts=artifacts, source_files=len(reviewed),
                installer_payload_files=len(payload_hashes), shared_files_checked=len(required),
                source_and_payload_checksums_match=True, platform_managers_identical=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('downloads', type=Path)
    parser.add_argument('--source', type=Path, required=True, help='Reviewed export with SOURCE-SHA256SUMS')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.downloads, args.source), indent=2))
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        sys.exit('Release verification failed: '+str(error))
