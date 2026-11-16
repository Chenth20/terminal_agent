"""Summarize the timed batches from benchmark.py."""
import argparse
import json
from pathlib import Path
from statistics import mean

from h100_config import PROFILES


def summarize(run):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    spec = json.loads((run / "inputs.json").read_text())
    profile = PROFILES[spec.get("profile", "baseline")]
    warmup = len(profile.warmup)
    expected = list(profile.timed)
    levels = sorted(set(profile.timed))
    end = warmup + len(expected)
    timings = rows[warmup:end]
    if [r["layers"] for r in timings] != expected:
        raise ValueError("Interleaved timing measurements are incomplete")
    if len({r["tokens"] for r in timings}) != 1:
        raise ValueError("Timed batches differ in token count")
    results = {}
    for layers in levels:
        pair = [r for r in timings if r["layers"] == layers]
        seconds = mean(r["seconds"] for r in pair)
        results[layers] = dict(seconds=seconds, tokens_per_second=pair[0]["tokens"]/seconds,
                               peak_allocated_GiB=max(r["allocated_GiB"] for r in pair))
    reference = 0 if 0 in results else 24
    for stats in results.values():
        stats[f"elapsed_increase_vs_{reference}_pct"] = 100*(stats["seconds"]/results[reference]["seconds"]-1)
        stats[f"throughput_loss_vs_{reference}_pct"] = 100*(1-stats["tokens_per_second"]/results[reference]["tokens_per_second"])
    return dict(timed=results, longest_batch=rows[end] if len(rows) > end else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
