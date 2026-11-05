"""Make a small, reviewable cluster report. Never export a raw log or chat.

The console and this command use the same allowlist. Adding a field is an
explicit decision: copying the state dictionary would also copy addresses,
hostnames, paths and conversation text.
"""
import argparse
import json
import math
import os
import platform
from pathlib import Path
import re
import subprocess
import urllib.error
import urllib.request


def number(value, fallback="unknown"):
    try:
        n = float(value)
        return f"{n:g}" if math.isfinite(n) and n >= 0 else fallback
    except (TypeError, ValueError, OverflowError):
        return fallback


def text(value):
    return re.sub(r"[\r\n|<>]", " ", str(value or "unknown"))[:120]


def local_hardware():
    """Called only when a person asks for a report, never during inference."""
    result = {"os": platform.system(), "arch": platform.machine(),
              "cores": os.cpu_count(), "cpu_model": platform.processor()}
    try:
        if platform.system() == "Darwin":
            def sysctl(key):
                return subprocess.check_output(["/usr/sbin/sysctl", "-n", key],
                                               text=True, timeout=3).strip()
            result.update(cpu_model=sysctl("machdep.cpu.brand_string"),
                          mem_total_bytes=int(sysctl("hw.memsize")))
        else:
            result["mem_total_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    result["cpu_model"] = line.partition(":")[2].strip()
                    break
    except (OSError, ValueError, subprocess.SubprocessError):
        pass  # Unknown is useful too; reporting must not require a benchmark.
    return result


def render(state, hardware=None, version="unknown"):
    nodes = state.get("nodes", {})
    names = ["head"] + [name for name in nodes if name != "head"]
    labels = {name: "Head" if name == "head" else f"Node {i}"
              for i, name in enumerate(names)}
    rows = ["# My Fronk cluster", "", f"Fronk version: {text(version)}",
            "Model: Qwen3.8-Flash-Next-Q4.gguf", "",
            "## Machines and allocation", "",
            "| Machine | CPU | RAM (GiB) | OS / architecture | Expert backend | RAM / SSD units | Link |",
            "|---|---|---:|---|---|---|---|"]
    for name in names:
        node = dict(nodes.get(name, {}))
        if name == "head":
            node.update(hardware or {})
        try:
            ram = f"{int(node['mem_total_bytes']) / 2**30:.1f}"
        except (KeyError, ValueError, TypeError):
            ram = "unknown"
        hybrid = node.get("hybrid", {})
        units = f"{number(hybrid.get('RAM', node.get('blocks')))} / {number(hybrid.get('SSD', 0))}"
        # Link type is a closed vocabulary; addresses and interface names stay local.
        link = node.get("link")
        link = link if link in ("thunderbolt", "ethernet", "wifi") else "local" if name == "head" else "unknown"
        rows.append(f"| {labels[name]} | {text(node.get('cpu_model'))} | {ram} | "
                    f"{text(node.get('os'))} / {text(node.get('arch'))} | "
                    f"{text(node.get('backend'))} | {units} | {link} |")
    rows += ["", "Dense head engine / GPU: [fill in; separate from its expert backend]",
             "Cables: [separate ports, or switch + head uplink speed + node port speeds]",
             "", "## Console test result", ""]
    report = state.get("report", {})
    for key, label in (("toks", "Reported decode rate (tok/s)"),
                       ("first_token", "Time to first token (s)"),
                       ("wall_s", "Whole request (s)"),
                       ("tokens", "Reported output count"),
                       ("deadline_misses", "Deadline misses")):
        rows.append(f"- {label}: {number(report.get(key))}")
    rows += [f"- Slowest participant: {labels.get(report.get('bottleneck'), 'unknown')}",
             "- Source: the console's test report; not a synthetic estimate or a guarantee.",
             "", "| Machine | Compute (µs) | Link/return (µs) | Last to reply (count) |",
             "|---|---:|---:|---:|"]
    for name in names:
        timing = nodes.get(name, {}).get("report", {})
        rows.append(f"| {labels[name]} | {number(timing.get('compute'))} | "
                    f"{number(timing.get('network'))} | {number(timing.get('last'))} |")
    rows += ["", "## What I tried", "", "Cold / warm run: [fill in]",
             "Prompt and context length: [a harmless reproduction, if sharing one]",
             "What worked, failed, or helped: [fill in]", "",
             "This file omits hostnames, addresses, pairing codes, paths and chat text.",
             "Review it, fill in what you know, and paste it into a GitHub Cluster result issue.",
             "Nothing has been uploaded. Unknown fields are fine.", ""]
    return "\n".join(rows)


def version_at(root):
    try:
        return re.search(r"^VERSION := (.+)$", (Path(root) / "Makefile").read_text(), re.M)[1]
    except (OSError, TypeError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console", default="http://127.0.0.1:7730")
    parser.add_argument("--output", default="cluster-result.md")
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(args.console.rstrip("/") + "/state", timeout=5) as response:
            state = json.load(response)
        report = render(state, local_hardware(), version_at(Path(__file__).resolve().parent.parent))
        # Do not overwrite somebody's completed report.
        with open(args.output, "x", encoding="utf-8") as out:
            out.write(report)
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.exit(1, f"Cannot save report: {error}\nOpen the head console, or choose another output filename.\n")
    print(f"Saved {args.output}. Review it before sharing; nothing was uploaded.")


if __name__ == "__main__":
    main()
