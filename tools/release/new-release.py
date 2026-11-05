#!/usr/bin/env python3
"""new-release.py - what changed since the last release, and who has to install it.

Fronk has one version and one set of packages: head and nodes run one binary from one
source tree, so a release is always complete. "A release of one part" means the notes
tell people which machines need it. Parts come from .github/modules.json, the same map
that labels pull requests. Run with no arguments to see the report; nothing is written.
--write VERSION sets the Makefile version and drafts docs/releases/VERSION.md.
Publishing stays a separate, deliberate step: see docs/RELEASE.md.
"""
import argparse
import fnmatch
import json
from pathlib import Path
import re
import subprocess
import sys

# Who has to install a change in each part. Anything not mapped is treated as "everyone".
WHO = {
    'console': 'the head',
    'head': 'the head',
    'backend': 'the nodes using that backend, and the head if it holds units',
    'planner': 'the head and every node',
    'installer': 'every machine, the next time its manager runs',
    'transport': 'the head and every node, together',
    'model': 'the head and every node, together',
}
NOBODY = {'docs', 'tests', 'build'}  # no machine runs these: merging them is enough
BREAKING = {'transport', 'model'}    # head and nodes must match: next minor version


def git(*args):
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout


def last_tag():
    tags = [t for t in git('tag', '--list', 'v[0-9]*', '--sort=-v:refname').split() if re.fullmatch(r'v\d+\.\d+\.\d+', t)]
    return tags[0] if tags else None


def parts_of(paths, rules):
    """Part -> changed files. A file no rule claims lands in 'other'."""
    out = {}
    for path in paths:
        names = [a for a, patterns in rules.items() if any(fnmatch.fnmatchcase(path, p) for p in patterns)]
        for name in names or ['other']: out.setdefault(name, []).append(path)
    return out


def suggest(version, parts):
    major, minor, patch = map(int, version.split('.'))
    if not set(parts)-NOBODY: return None
    if set(parts) & BREAKING: return f'{major}.{minor+1}.0'
    return f'{major}.{minor}.{patch+1}'


def who(parts):
    running = sorted(set(parts)-NOBODY)
    if 'other' in running: return ['every machine (some files belong to no part: check them)']
    return [f'{name}: {WHO[name]}' for name in running]


def notes(version, since, parts):
    running = sorted(set(parts)-NOBODY) or sorted(parts)
    lines = [f'# Fronk {version}: {", ".join(running)}', '', 'Replace this line with the change in one or two plain sentences.', '',
             '## Who needs to update', '']
    lines += [f'- {line}' for line in who(parts)] or ['- Nobody needs to update.']
    lines += ['', 'Other machines can stay on their version unless listed above. To update, open the new',
              'manager on the machine and choose upgrade: saved configuration and verified weights are kept.', '',
              '## What changed', '']
    for name in sorted(parts):
        subjects = git('log', '--format=%s', f'{since}..HEAD', '--', *parts[name]).split('\n')
        lines += [f'- {name}: {s}' for s in subjects if s]
    lines += ['', '## What was tested', '', 'Commands, results and hardware. Name what was not tested.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--since', help='tag to compare with (default: the newest vX.Y.Z)')
    ap.add_argument('--write', metavar='VERSION', help='set this version and draft its release notes')
    args = ap.parse_args()
    root = Path(git('rev-parse', '--show-toplevel').strip())
    since = args.since or last_tag()
    if not since: sys.exit('No vX.Y.Z tag yet: tag the first release, then this can compare against it.')
    makefile = root/'Makefile'
    version = re.search(r'^VERSION := (.+)$', makefile.read_text(), re.M)[1]
    paths = [p for p in git('diff', '--name-only', f'{since}..HEAD').split('\n') if p]
    parts = parts_of(paths, json.loads((root/'.github/modules.json').read_text()))
    next_version = suggest(version, parts)

    print(f'Since {since}: {len(paths)} files in {len(parts)} parts')
    for name in sorted(parts): print(f'  {name:10} {len(parts[name])} files')
    if not next_version:
        print('No machine runs these files: merge them, no release is needed.')
        if not args.write: return
    else:
        print('Who has to update:')
        for line in who(parts): print(f'  {line}')
        print(f'Suggested version: {next_version}')
    if not args.write:
        print(f'Next: python3 tools/release/new-release.py --write {next_version}')
        return

    if not re.fullmatch(r'\d+\.\d+\.\d+', args.write): sys.exit('VERSION must look like 0.3.1')
    draft = root/'docs/releases'/f'{args.write}.md'
    if draft.exists(): sys.exit(f'{draft.relative_to(root)} already exists: nothing changed.')
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_text(notes(args.write, since, parts))
    makefile.write_text(re.sub(r'^VERSION := .+$', f'VERSION := {args.write}', makefile.read_text(), flags=re.M))
    print(f'Wrote {draft.relative_to(root)} and set VERSION := {args.write}. Edit the notes, then follow docs/RELEASE.md.')


if __name__ == '__main__': main()
