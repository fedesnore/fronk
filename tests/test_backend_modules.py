"""Manifest/build boundaries; real compilation is in backend-modules.sh."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('modules', ROOT/'tools/backend-module.py')
modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modules)


class Modules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.module = self.root/'backends/modules/tinycpu'
        modules.new_module('tinycpu', self.module)

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, **fields):
        p = self.module/'backend.json'
        data = json.loads(p.read_text()); data.update(fields)
        p.write_text(json.dumps(data))

    def test_template_and_registration(self):
        modules.configure(self.root, 'Linux', 'x86_64')
        header = self.root/'build/backend-modules/registry.h'
        self.assertIn('FR_BACKEND_MODULE(tinycpu)', header.read_text())
        self.assertIn('fr_module_tinycpu', (self.module/'backend.c').read_text())
        before = header.stat().st_mtime_ns
        modules.configure(self.root, 'Linux', 'x86_64')
        self.assertEqual(before, header.stat().st_mtime_ns)
        self.edit(systems=['Darwin'])
        modules.configure(self.root, 'Linux', 'x86_64')
        self.assertNotIn('FR_BACKEND_MODULE(', header.read_text())

    def test_rejects_invalid_contracts(self):
        original = (self.module/'backend.json').read_text()
        for fields in [dict(api=True), dict(api=2), dict(name='metal'), dict(name='ref_fast'),
                       dict(name='wrong-name'), dict(name='x'*16), dict(sources=['../escape.c']),
                       dict(sources=['missing.c']), dict(sources=['backend.c']*2),
                       dict(ldlibs=['$(shell bad)']), dict(systems=['Other']), dict(extra=1)]:
            with self.subTest(fields=fields):
                (self.module/'backend.json').write_text(original)
                self.edit(**fields)
                with self.assertRaises(ValueError): modules.read_module(self.module)

    def test_rejects_symlinks(self):
        (self.module/'outside.h').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError): modules.read_module(self.module)

    def test_copy_and_replacement_preserve_source(self):
        dest = self.root/'saved'
        modules.copy_module(self.module, dest, False)
        modules.copy_module(self.module, dest, False)  # An identical upgrade is harmless.
        old = (dest/'tinycpu/backend.c').read_bytes()
        (self.module/'backend.c').write_bytes(old+b'\n/* revision */\n')
        with self.assertRaises(ValueError): modules.copy_module(self.module, dest, False)
        self.assertEqual(old, (dest/'tinycpu/backend.c').read_bytes())
        modules.copy_module(self.module, dest, True)
        self.assertEqual(old, (dest/'.tinycpu.previous/backend.c').read_bytes())
        self.assertEqual((self.module/'backend.c').read_bytes(), (dest/'tinycpu/backend.c').read_bytes())

    def test_names_and_limit(self):
        self.edit(name='different')
        with self.assertRaises(ValueError): modules.configure(self.root, 'Linux', 'x86_64')
        self.edit(name='tinycpu')
        for n in range(modules.MAX_MODULES):
            modules.new_module('cpu'+str(n), self.root/'backends/modules'/('cpu'+str(n)))
        with self.assertRaises(ValueError): modules.configure(self.root, 'Linux', 'x86_64')


if __name__ == '__main__': unittest.main()
