#!/usr/bin/env python3
"""Read saved QA only; never contact, reconfigure or stop the cluster.

Engine cycles and HTTP fragments are intentionally reported separately.
The raw inputs are hashed so a result can be tied to the exact saved logs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median


def summarize(qa):
    inputs = {}

    def read(name):
        raw = (qa / name).read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    perf = json.loads(read("native-mtp-results.json"))
    result = {"method": {
        "cycles": "Drop the first request; omit T=1 startup cycles from remaining requests.",
        "http": "Historical harness counts SSE text fragments, not independently tokenized output.",
        "scope": "One short greedy prose prompt, three warm repetitions; no general workload claim.",
        "limits": "Sequential depth tests, no interleaved SSD control; aggregate logs do not isolate GPU vs network vs each node.",
    }, "depths": {}}
    for depth in (4, 6, 8):
        text = read(f"native-depth{depth}.log")
        cycles = [dict(re.findall(r"(\w+)=([0-9.]+)", line))
                  for line in re.findall(r"FRONK_NATIVE depth=[^\n]+", text)]
        starts = [i for i, cycle in enumerate(cycles) if int(cycle["T"]) == 1]
        if len(starts) < 2:
            raise ValueError(f"Depth {depth}: no complete warm request boundary")
        warm = [c for c in cycles[starts[1]:] if int(c["T"]) > 1]
        if not warm or any(int(c["T"]) != depth for c in warm):
            raise ValueError(f"Depth {depth}: missing or mixed-width warm cycles")
        times = {key: mean(float(c[key]) for c in warm)
                 for key in ("verify_ms", "commit_ms", "draft_ms")}
        accepted = mean(int(c["accepted"]) for c in warm)
        recovery = [float(c["commit_ms"]) for c in warm if int(c["replay"]) > 0]
        checks = read(f"native-depth{depth}-check.log")
        if " FAIL" in checks:
            raise ValueError(f"Depth {depth}: failed correctness check in input")
        result["depths"][str(depth)] = {
            "warm_cycles": len(warm),
            "accepted_mean": accepted,
            "accepted_histogram": dict(sorted(Counter(int(c["accepted"]) for c in warm).items())),
            "mean_ms": times,
            "cycle_ms": sum(times.values()),
            "engine_cycle_tokens_per_second": accepted * 1000 / sum(times.values()),
            "http_fragments_per_second_median": median(p["tps"] for p in perf[f"depth{depth}"]["warm"]),
            "cycle_budget_ms_at_current_acceptance": {str(tps): accepted * 1000 / tps for tps in (16, 30)},
            "recovery_events": len(recovery),
            "recovery_ms_mean": mean(recovery) if recovery else 0,
            "logit_checks": checks.count("FRONK_NATIVE_CHECK"),
            "state_checks": checks.count("FRONK_NATIVE_STATE"),
        }
    baseline = json.loads(read("combined-ssd-results.json"))
    result["historical_ssd_http_fragments_per_second"] = {
        label: [p["tps"] for p in baseline[label]["warm"]]
        for label in ("ssd-before", "ssd-after")
    }
    result["input_sha256"] = inputs
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path(__file__).resolve().parents[2] / "build/revision-qa")
    args = parser.parse_args()
    print(json.dumps(summarize(args.qa), indent=2, allow_nan=False))
