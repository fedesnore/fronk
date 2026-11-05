"""Check local guide links and heading anchors; no network or dependencies.

Keep this usable both in a source checkout and in the installed runtime. A link
to an existing file is still broken if its heading was renamed.
"""
from pathlib import Path
from html import unescape
import re
import sys
from urllib.parse import unquote


def anchors(text):
    result = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    counts = {}
    fence = None
    for line in text.splitlines():
        code = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if code:
            marker = code[1][0]
            if fence is None: fence = marker
            elif marker == fence: fence = None
            continue
        if fence: continue
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not heading: continue
        title = re.sub(r'<[^>]+>', '', unescape(heading[1])).lower()
        title = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', title)
        slug = re.sub(r'[^\w\- ]', '', title).replace(' ', '-')
        occurrence = counts.get(slug, 0)
        counts[slug] = occurrence + 1
        result.add(slug + (f'-{occurrence}' if occurrence else ''))
    return result

root = Path(__file__).resolve().parent.parent
names = ["README.md", "AGENTS.md", "docs/README.md", "PROJECT.md", "CONTRIBUTING.md", "BACKENDS.md", "BENCH.md", "RUN.md",
         "THIRD_PARTY.md", "docs/INSTALL.md", "docs/MODEL.md", "docs/NETWORK.md",
         "docs/PROTOCOL.md", "docs/RELEASE.md", "docs/TROUBLESHOOTING.md",
         "docs/MEASUREMENTS.md", "docs/SHARE.md", "docs/IMPROVEMENTS.md", "docs/BACKEND_MODULES.md", "docs/MODULES.md",
         "docs/API.md", "docs/TEST_CLUSTER.md", "docs/ABOUT.md", "docs/PROJECTIONS.md"]
names += [str(p.relative_to(root)) for folder in ("docs/llm", "docs/evidence", "docs/releases")
          for p in sorted((root / folder).glob("*.md"))]
names += ["docs/PLAN-mtp-native.md"]
errors = []
for name in names:
    path = root / name
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
        if target.startswith(("http:", "https:", "mailto:")):
            continue
        relative, _, fragment = unquote(target).partition("#")
        destination = path.parent / relative if relative else path
        if not destination.exists():
            errors.append(f"{name}: missing local link {target}")
        elif fragment and destination.suffix == '.md' and fragment not in anchors(destination.read_text()):
            errors.append(f"{name}: missing heading {target}")
if errors:
    sys.exit("\n".join(errors))
print(f"PASS: local links in {len(names)} public guides")
