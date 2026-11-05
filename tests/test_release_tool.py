"""new-release.py and pr-scope.py against a throwaway Git repository."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class ReleaseTool(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(); self.repo = Path(self.dir.name)
        (self.repo/'.github').mkdir(); (self.repo/'src').mkdir(); (self.repo/'docs').mkdir(); (self.repo/'tools').mkdir()
        for name in ('modules.json', 'triage.py'): shutil.copy(ROOT/'.github'/name, self.repo/'.github'/name)
        (self.repo/'Makefile').write_text('# build\nVERSION := 0.3.0\n')
        self.git('init', '-q', '-b', 'main'); self.git('config', 'user.name', 'T'); self.git('config', 'user.email', 't@example.invalid')
        self.commit('first'); self.git('tag', 'v0.3.0')

    def tearDown(self): self.dir.cleanup()

    def git(self, *args): subprocess.run(['git', *args], cwd=self.repo, check=True, capture_output=True)

    def commit(self, message): self.git('add', '-A'); self.git('commit', '-q', '-m', message)

    def tool(self, *args, script='tools/release/new-release.py'):
        return subprocess.run([sys.executable, str(ROOT/script), *args], cwd=self.repo, capture_output=True, text=True)

    def change(self, path, message='change'):
        (self.repo/path).parent.mkdir(parents=True, exist_ok=True); (self.repo/path).write_text('x\n'); self.commit(message)

    def test_docs_need_no_release(self):
        self.change('docs/GUIDE.md')
        self.assertIn('no release is needed', self.tool().stdout)

    def test_one_part_names_its_machines_and_a_patch(self):
        self.change('tools/fronk-web.py', 'console: clearer status')
        out = self.tool().stdout
        self.assertIn('console: the head', out); self.assertIn('Suggested version: 0.3.1', out)
        self.assertNotIn('transport', out)

    def test_protocol_change_asks_for_a_minor_and_all_machines(self):
        self.change('src/plane.c')
        out = self.tool().stdout
        self.assertIn('Suggested version: 0.4.0', out); self.assertIn('every node, together', out)

    def test_unknown_files_mean_everyone(self):
        self.change('new-thing/file.c')
        self.assertIn('every machine', self.tool().stdout)

    def test_write_sets_version_and_never_overwrites_notes(self):
        self.change('tools/fronk-web.py', 'console: clearer status')
        result = self.tool('--write', '0.3.1'); self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('VERSION := 0.3.1', (self.repo/'Makefile').read_text())
        notes = (self.repo/'docs/releases/0.3.1.md').read_text()
        self.assertIn('# Fronk 0.3.1: console', notes); self.assertIn('console: clearer status', notes)
        again = self.tool('--write', '0.3.1'); self.assertNotEqual(again.returncode, 0)
        self.assertEqual(notes, (self.repo/'docs/releases/0.3.1.md').read_text())

    def test_pr_scope_encourages_one_part(self):
        self.git('checkout', '-q', '-b', 'one'); self.change('tools/fronk-web.py'); self.change('docs/A.md')
        self.assertIn('One part: good', self.tool('main', script='tools/pr-scope.py').stdout)
        self.git('checkout', '-q', '-b', 'two'); self.change('src/plane.c')
        self.assertIn('one pull request', self.tool('main', script='tools/pr-scope.py').stdout)


if __name__ == '__main__': unittest.main()
