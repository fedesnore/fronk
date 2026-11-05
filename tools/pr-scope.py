#!/usr/bin/env python3
"""pr-scope.py - which parts of Fronk this branch touches, before opening a pull request.

Prints the labels the repository will give the pull request, using the same rules as
.github/triage.py. One part at a time is what we ask for: it is reviewed and accepted
on its own. Usage: python3 tools/pr-scope.py [base branch, default main]
"""
import importlib.util
from pathlib import Path
import subprocess
import sys


def git(*args):
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout


root = Path(git('rev-parse', '--show-toplevel').strip())
spec = importlib.util.spec_from_file_location('triage', root/'.github/triage.py')
triage = importlib.util.module_from_spec(spec); spec.loader.exec_module(triage)
base = sys.argv[1] if len(sys.argv) > 1 else 'main'
merge = git('merge-base', base, 'HEAD').strip()
paths = sorted({p for p in git('diff', '--name-only', merge).split('\n') + git('ls-files', '--others', '--exclude-standard').split('\n') if p})
if not paths: sys.exit(f'No changes against {base}.')
labels = sorted(triage.classify(paths))
print(f'{len(paths)} files. Labels: ' + (', '.join(labels) or 'none: these files belong to no part, say where they fit'))
if 'scope:cross-module' in labels:
    print('This touches more than one part. If the parts can land separately, make one pull request')
    print('for each. If they cannot, say why in the pull request: docs/MODULES.md.')
else:
    print('One part: good. Docs and tests that go with it do not count as another part.')
