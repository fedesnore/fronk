#!/usr/bin/env python3
"""Summarize saved complete AVX2 request replays; never operates the cluster."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def analyze(qa):
    inputs, nodes = {}, {}

    def read(name):
        raw = (qa / name).read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    routes = json.loads(read("avx-route-reuse.json"))
    for node in ("imac", "shuttle"):
        log = read(f"avx-mtp-{node}-combined.log")
        if "PASS all captured requests bit-identical" not in log:
            raise ValueError(f"Missing exact-output qualification: {node}")
        shape = re.search(r"BACKEND workers=(\d+) blocks=(\d+) real_units=([\d,]+)", log)
        capture = re.search(r"REPLAY calls=(\d+) token_rows=(\d+) minimum_shared_rows=100", log)
        matches = re.findall(r"REPLAY repeat=(\d+) before_us=([\d.]+) candidate_us=([\d.]+) after_us=([\d.]+)", log)
        if not shape or not capture or len(matches) != 5 or [int(r[0]) for r in matches] != list(range(5)):
            raise ValueError(f"Incomplete combined A/B/A replay: {node}")
        if int(capture[1]) != routes["records"]:
            raise ValueError(f"Routing count does not match replay: {node}")
        rows = [{"before_us": float(b), "candidate_us": float(c), "after_us": float(a)}
                for _, b, c, a in matches]
        medians = {key: median(row[key] for row in rows) for key in rows[0]}
        diagnostic = re.search(r"diagnostic_only raw_compute_us=([\d.]+) ordered_reduce_us=([\d.]+)", log)
        nodes[node] = {
            "workers": int(shape[1]), "blocks": int(shape[2]),
            "unit_ids": [int(s) for s in shape[3].split(",")],
            "requests": int(capture[1]), "token_rows": int(capture[2]),
            "exact_outputs": True, "repeats": rows, "median_us": medians,
            "median_time_reduction_percent_vs_controls": {
                phase: 100 * (1 - medians["candidate_us"] / medians[phase + "_us"])
                for phase in ("before", "after")},
            "median_paired_time_reduction_percent": median(
                100 * (1 - r["candidate_us"] / ((r["before_us"] + r["after_us"]) / 2)) for r in rows),
            "separate_diagnostic_us": None if not diagnostic else {
                "raw_compute": float(diagnostic[1]), "ordered_reduce": float(diagnostic[2])},
        }
    return {
        "scope": "Resident native-unit backend replay, T=8. Includes grouping, quantization, pool dispatch, compute, ordered reduction and final output. Excludes network, dense head, SSD streaming, proposal and acceptance. No end-to-end tokens/s claim.",
        "controls": "Same process, weights and recorded rounded activations; row kernels and cross-rank union off/on/off. Diagnostic phase timers off during A/B/A.",
        "routing": {**routes, "logical_group_reduction_percent": 100 * (1 - routes["union_mean"] / routes["rank_mean"])},
        "nodes": nodes, "input_sha256": inputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path(__file__).resolve().parents[2] / "build/revision-qa")
    args = parser.parse_args()
    print(json.dumps(analyze(args.qa), indent=2, allow_nan=False))
