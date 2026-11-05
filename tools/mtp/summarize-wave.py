#!/usr/bin/env python3
"""Analyze a saved head wave off/on/off trial, without touching the cluster."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median


def analyze(qa, depth):
    tag = f"wave-ab-matrix1-d{depth}-bench"
    hashes, phases, identities = {}, {}, {}

    def read(name):
        data = (qa / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data.decode()

    http = json.loads(read(tag + "-results.json"))
    hybrid = {}
    for phase in ("before", "candidate", "after"):
        log = read(tag + "-" + phase + ".log")
        hybrid[phase] = re.findall(
            r"HYBRID node=(\S+) RAM=(\d+) SSD=(\d+) cache=(\d+) MiB burst=(\d+)[^\n]*ssd_mask=(\d+)", log)
        if "TIMING READY " not in log:
            raise ValueError("Missing warm boundary: " + phase)
        warm = log.split("TIMING READY ", 1)[1]
        cycles = [dict(re.findall(r"(\w+)=([^\s]+)", line))
                  for line in re.findall(r"FRONK_NATIVE depth=[^\n]+", warm)]
        if not cycles or any("wave" not in c or "draft_hash" not in c for c in cycles):
            raise ValueError("Missing wave/cycle identity: " + phase)
        for c in cycles:
            if int(c["wave"]) != int(phase == "candidate" and int(c["T"]) >= 4):
                raise ValueError("Unexpected wave policy: " + phase)
        identities[phase] = [(c["T"], c["accepted"], c["replay"], c["draft_hash"]) for c in cycles]
        grouped = defaultdict(list)
        for c in cycles:
            grouped[int(c["T"])].append(c)
        times = [{"T": t, "cycles": len(cs),
                  "accepted_mean": mean(int(c["accepted"]) for c in cs),
                  **{key: mean(float(c[key]) for c in cs)
                     for key in ("verify_ms", "commit_ms", "draft_ms")}}
                 for t, cs in sorted(grouped.items())]
        samples = http[phase]["samples"]
        cycle_ms = sum(float(c[k]) for c in cycles for k in ("verify_ms", "commit_ms", "draft_ms"))
        phases[phase] = {
            "samples": [{k: s[k] for k in ("prompt", "decode_tps", "completion_tps", "first_s", "total_s", "usage")} for s in samples],
            "decode_tps_median": median(s["decode_tps"] for s in samples),
            "cycle_tokens_per_second": 1000 * sum(int(c["accepted"]) for c in cycles) / cycle_ms,
            "cost_by_width": times,
        }
    comparisons = {}
    for phase in ("candidate", "after"):
        a, b = http["before"]["samples"], http[phase]["samples"]
        comparisons[phase] = {
            "same_text_token_count_and_prompts": len(a) == len(b) and all(
                x["text"] == y["text"] and x["usage"]["completion_tokens"] == y["usage"]["completion_tokens"] and
                x["prompt"] == y["prompt"] for x, y in zip(a, b)),
            "same_unit_placement": http[phase]["placement"] == http["before"]["placement"],
            "same_hybrid_masks_and_cache_budgets": hybrid[phase] == hybrid["before"],
            "same_width_acceptance_recovery_drafts": identities[phase] == identities["before"],
            "warm_cycles": len(identities[phase]),
        }
    return {"scope": "Three prompts, twice each. Exclude cold startup from cycle data; API usage supplies throughput counts. A/B/A uses the same native matrix arithmetic and proposer, changing only two-group verification. No diagnostic oracle or GPU encoder trace during timing.",
            "phases": phases, "comparisons": comparisons, "input_sha256": hashes}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qa", type=Path, default=Path(__file__).resolve().parents[2] / "build/revision-qa")
    p.add_argument("--depth", type=int, choices=range(4, 9), default=8)
    args = p.parse_args()
    print(json.dumps(analyze(args.qa, args.depth), indent=2, allow_nan=False))
