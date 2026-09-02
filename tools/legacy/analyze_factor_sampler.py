"""Report one deterministic epoch of tempered factor-balanced sampling."""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from samplers import DistributedFactorBalancedSampler


class Rows:
    def __init__(self, rows):
        self.samples = rows

    def __len__(self):
        return len(self.samples)


def quantiles(counter):
    values = sorted(counter.values())

    def q(frac):
        return values[min(len(values) - 1, int((len(values) - 1) * frac))]

    return {"min": values[0], "p10": q(.1), "p50": q(.5),
            "p90": q(.9), "max": values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="5script/train.csv")
    parser.add_argument("--char-alpha", type=float, default=.5)
    parser.add_argument("--callig-alpha", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=224)
    parser.add_argument("--cumulative-steps", default="",
                        help="Comma-separated optimizer steps for exact cumulative row coverage.")
    args = parser.parse_args()
    with open(args.csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sampler = DistributedFactorBalancedSampler(
        Rows(rows), seed=args.seed, char_alpha=args.char_alpha,
        callig_alpha=args.callig_alpha)
    indices = list(sampler)
    chosen = [rows[index] for index in indices]
    result = {
        "rows": len(rows),
        "unique_rows_drawn": len(set(indices)),
        "unique_row_fraction": len(set(indices)) / len(rows),
        "weights": sampler.summary(),
        "source": {
            "char": quantiles(Counter(r["character_id"] for r in rows)),
            "callig": quantiles(Counter(r["calligrapher_id"] for r in rows)),
            "script": dict(Counter(r["script_id"] for r in rows)),
        },
        "sampled": {
            "char": quantiles(Counter(r["character_id"] for r in chosen)),
            "callig": quantiles(Counter(r["calligrapher_id"] for r in chosen)),
            "script": dict(Counter(r["script_id"] for r in chosen)),
        },
    }
    if args.cumulative_steps:
        requested = sorted({int(value) for value in args.cumulative_steps.split(",")})
        steps_per_epoch = len(rows) // args.batch
        exposure = torch.zeros(len(rows), dtype=torch.int64)
        coverage = {}
        completed_steps = 0
        epoch = 0
        while requested and completed_steps < requested[-1]:
            sampler.set_epoch(epoch)
            epoch_indices = list(sampler)[:steps_per_epoch * args.batch]
            steps_here = min(steps_per_epoch, requested[-1] - completed_steps)
            consumed = 0
            for target_step in [step for step in requested
                                if completed_steps < step <= completed_steps + steps_here]:
                take = (target_step - completed_steps) * args.batch
                segment = torch.as_tensor(epoch_indices[consumed:take], dtype=torch.int64)
                exposure += torch.bincount(segment, minlength=len(rows))
                consumed = take
                sorted_exposure = exposure.sort().values
                unique_rows = int((exposure > 0).sum())
                coverage[str(target_step)] = {
                    "dataset_equivalents": target_step * args.batch / len(rows),
                    "unique_rows_seen": unique_rows,
                    "unique_row_fraction": unique_rows / len(rows),
                    "rows_never_seen": len(rows) - unique_rows,
                    "exposure_min": int(sorted_exposure[0]),
                    "exposure_p01": int(sorted_exposure[int(.01 * (len(rows) - 1))]),
                    "exposure_p10": int(sorted_exposure[int(.10 * (len(rows) - 1))]),
                    "exposure_median": int(sorted_exposure[len(rows) // 2]),
                }
            final_take = steps_here * args.batch
            if consumed < final_take:
                segment = torch.as_tensor(epoch_indices[consumed:final_take], dtype=torch.int64)
                exposure += torch.bincount(segment, minlength=len(rows))
            completed_steps += steps_here
            epoch += 1
        result["cumulative_coverage"] = coverage
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
