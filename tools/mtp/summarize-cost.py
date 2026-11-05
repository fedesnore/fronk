#!/usr/bin/env python3
"""Check fixed placement/output and report both controls, without averaging away drift."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(path):
    raw = path.read_bytes()
    runs = json.loads(raw)
    reference = runs["before"]
    base_samples = reference["samples"]
    if not base_samples or "after" not in runs:
        raise ValueError("Incomplete bracketed experiment")
    medians = {}
    for name, run in runs.items():
        if run["placement"] != reference["placement"]:
            raise ValueError("Placement changed: " + name)
        samples = run["samples"]
        if len(samples) != len(base_samples):
            raise ValueError("Incomplete samples: " + name)
        for a, b in zip(base_samples, samples):
            if any(a[k] != b[k] for k in ("prompt", "text")) or a["usage"]["completion_tokens"] != b["usage"]["completion_tokens"]:
                raise ValueError("Output/workload changed: " + name)
            if not math.isfinite(b["decode_tps"]) or b["decode_tps"] <= 0:
                raise ValueError("Invalid throughput")
        medians[name] = statistics.median(s["decode_tps"] for s in samples)
    effects = {}
    for name, value in medians.items():
        if name in ("before", "after"):
            continue
        effects[name] = {
            "versus_before_percent": 100 * (value / medians["before"] - 1),
            "versus_after_percent": 100 * (value / medians["after"] - 1),
            "exceeds_both_controls": value > max(medians["before"], medians["after"]),
        }
    return {
        "input": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "method": "Three prompts twice, API completion tokens, identical text/count/placement; both surrounding controls reported. These small samples do not establish statistical significance.",
        "samples_per_phase": len(base_samples), "median_decode_tps": medians,
        "effects": effects, "placement": reference["placement"],
    }


if __name__ == "__main__":
    print(json.dumps([summarize(Path(name)) for name in sys.argv[1:]], indent=2))
