#!/usr/bin/env python3
"""Analyze a saved native catch-up off/on/off trial; no cluster actions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median


def analyze(qa):
    inputs, phases, sequences = {}, {}, {}

    def read(name):
        raw = (qa / name).read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    prefix = "catchup-matrix1-d4-bench"
    http = json.loads(read(prefix + "-results.json"))
    for phase in ("before", "candidate", "after"):
        text = read(prefix + "-" + phase + ".log")
        if "TIMING READY " not in text:
            raise ValueError("Missing startup boundary: " + phase)
        text = text.split("TIMING READY ", 1)[1]
        cycles = [dict(re.findall(r"(\w+)=([^ ]+)", line))
                  for line in re.findall(r"FRONK_NATIVE depth=[^\n]+", text)]
        if not cycles or any("draft_hash" not in c for c in cycles):
            raise ValueError("Missing proposal identity: " + phase)
        sequences[phase] = [(c["T"], c["accepted"], c["replay"], c["draft_hash"]) for c in cycles]
        grouped = defaultdict(list)
        for c in cycles:
            grouped[(int(c["T"]), int(c["accepted"]))].append(c)
        costs = []
        for (width, accepted), rows in sorted(grouped.items()):
            costs.append({"T": width, "accepted": accepted, "cycles": len(rows),
                          "skipped_heads": sum(int(r["skipped_heads"]) for r in rows),
                          **{key: mean(float(r[key]) for r in rows)
                             for key in ("verify_ms", "commit_ms", "draft_ms", "catchup_ms", "chain_ms")}})
        phases[phase] = {"costs": costs, "samples": [
            {key: s[key] for key in ("decode_tps", "completion_tps", "first_s", "total_s", "usage")}
            for s in http[phase]["samples"]]}
        phases[phase]["decode_tps_median"] = median(s["decode_tps"] for s in http[phase]["samples"])
    comparisons = {}
    for phase in ("candidate", "after"):
        pairs = list(zip(http["before"]["samples"], http[phase]["samples"]))
        comparisons[phase] = {
            "same_cycle_width_acceptance_replay_and_draft_hash": sequences[phase] == sequences["before"],
            "warm_cycles": len(sequences[phase]),
            "same_http_text_and_completion_count": len(pairs) == len(http["before"]["samples"]) and all(
                a["text"] == b["text"] and a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"]
                for a, b in pairs),
            "same_placement": http[phase]["placement"] == http["before"]["placement"],
        }
    return {"method": "Drop cold startup. Compare catch-up at equal T and accepted prefix. API usage supplies throughput counts; no encoder trace/oracle/path profiling in timed phases.",
            "phases": phases, "comparisons": comparisons, "input_sha256": inputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path(__file__).resolve().parents[2] / "build/revision-qa")
    args = parser.parse_args()
    print(json.dumps(analyze(args.qa), indent=2, allow_nan=False))
