#!/usr/bin/env python3
"""Rank saved diagnostic costs by width; never contact or alter the cluster.

Serial plane stages can be summed. Node arrivals overlap each other and the
local backend, and the worker encloses the plane; neither can be added again.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def summarize(path):
    raw = path.read_bytes()
    text = raw.decode()
    # Exclude the cold startup request. Counters flush in 480-call windows,
    # so the first subsequent window may straddle this boundary.
    if "TIMING READY " in text:
        text = text.split("TIMING READY ", 1)[1]
    groups = defaultdict(lambda: {"calls": 0, "sum": defaultdict(float)})
    for line in text.splitlines():
        match = re.search(r"FRONK_PATH(_NODE|_WORKER)? (.*)", line)
        if not match:
            continue
        fields = dict(re.findall(r"(\w+)=([^ ]+)", match[2]))
        key = (match[1] or "_PLANE", int(fields["T"]), fields.get("node", ""))
        group = groups[key]
        count = int(fields["calls"])
        group["calls"] += count
        for name, value in fields.items():
            if name.endswith("_us"):
                group["sum"][name] += float(value) * count
            elif name == "last":
                group["sum"]["last_count"] += int(value)
    result = []
    for (kind, width, node), group in sorted(groups.items()):
        count = group["calls"]
        row = {"kind": kind[1:].lower(), "T": width, "calls": count}
        if node:
            row["node"] = node
        row.update({key: value if key == "last_count" else value / count
                    for key, value in group["sum"].items()})
        if kind == "_PLANE":
            names = ("prepare_us", "submit_us", "send_us", "wait_us", "finish_us")
            row["total_us"] = sum(row[key] for key in names)
            row["serial_stage_share"] = {key: row[key] / row["total_us"] for key in names}
        if kind == "_NODE":
            row["last_fraction"] = row["last_count"] / count
        result.append(row)
    if not result:
        raise ValueError(f"No path diagnostics in {path}")
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "costs": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps({"method": {
        "scope": "Diagnostic builds, weighted by calls and separated by T; startup section excluded.",
        "coverage": "480-call flush windows; first warm window may include startup tail; final incomplete window omitted.",
        "clock": "All path intervals use the head monotonic clock. Worker ready is a CPU wait, not GPU hardware time.",
        "overlap": "Plane stage sums are serial. Local compute and node arrivals overlap; worker moe encloses the plane.",
        "performance": "Repeat with diagnostic macro absent before claiming application speedup."
    }, "runs": [summarize(p) for p in args.logs]}, indent=2, allow_nan=False))
