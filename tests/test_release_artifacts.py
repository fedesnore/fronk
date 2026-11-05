"""A valid outer checksum must not hide missing, changed or unsafe payloads."""
import importlib.util
import io
from pathlib import Path
import tarfile
import unittest

spec = importlib.util.spec_from_file_location('artifacts', Path(__file__).resolve().parents[1]/'tools/release/verify-artifacts.py')
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class Artifacts(unittest.TestCase):
    def test_inventory_requires_exact_files_and_bytes(self):
        original = {'src/main.c': b'original'}
        original['SOURCE-SHA256SUMS'] = (artifacts.sha(b'original')+'  src/main.c\n').encode()
        artifacts.check_inventory(original)
        for change in [{'src/main.c': b'changed'}, {'extra.c': b'unlisted'}]:
            with self.assertRaises(ValueError): artifacts.check_inventory({**original, **change})
        with self.assertRaises(ValueError):
            artifacts.check_inventory({'SOURCE-SHA256SUMS': original['SOURCE-SHA256SUMS']})

    def test_rejects_duplicate_and_unsafe_checksum_paths(self):
        digest = '0'*64
        for text in [f'{digest}  file\n{digest}  file\n', f'{digest}  ../file\n', f'{digest}  /file\n']:
            with self.assertRaises(ValueError): artifacts.manifest(text)

    def test_archive_rejects_symlink_and_traversal(self):
        for name, kind in [('fronk/link', tarfile.SYMTYPE), ('fronk/../escape', tarfile.REGTYPE)]:
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w:gz') as tar:
                info = tarfile.TarInfo(name); info.type = kind; info.linkname = '/etc/passwd'
                tar.addfile(info)
            with self.assertRaises(ValueError): artifacts.archive(stream.getvalue(), 'fronk')


if __name__ == '__main__': unittest.main()
