#!/usr/bin/env python3
"""Bootstrap completion periods from observed, comparable throughput periods.

CSV input must contain period and completed columns, including zero-work periods.
This is a conditional scenario model, not a calibrated completion guarantee.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path


def forecast(throughput, remaining, trials=10000, max_periods=1000, seed=0):
    if not throughput or any(type(n) is not int or n < 0 for n in throughput):
        raise ValueError("throughput must be nonempty nonnegative integer counts")
    if type(remaining) is not int or remaining < 0:
        raise ValueError("remaining must be a nonnegative integer count")
    if type(trials) is not int or type(max_periods) is not int or trials < 1 or max_periods < 1:
        raise ValueError("trials and max_periods must be positive")
    if remaining and not any(throughput):
        raise ValueError("all observed periods have zero throughput; no completion forecast")
    rng = random.Random(seed)
    durations = []
    for _ in range(trials):
        left, periods = remaining, 0
        while left > 0 and periods < max_periods:
            left -= rng.choice(throughput)
            periods += 1
        # Keep censored trials in the denominator instead of dropping failures.
        durations.append(periods if left <= 0 else max_periods + 1)
    durations.sort()
    def quantile(q):
        value = durations[max(0, math.ceil(q * trials) - 1)]
        return value if value <= max_periods else None
    return {
        "p50_periods": quantile(.50), "p85_periods": quantile(.85),
        "censored_trials": sum(n > max_periods for n in durations),
        "trials": trials, "max_periods": max_periods, "seed": seed,
        "observed_periods": len(throughput), "throughput": throughput,
        "remaining_units": remaining,
        "assumptions": "Comparable periods/work mix; independent stationary resampling; fixed scope/capacity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--remaining", type=int, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--max-periods", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        with args.csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"period", "completed"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV needs period and completed columns")
            rows = list(reader)
        periods = [row["period"] for row in rows]
        if any(not period.strip() for period in periods) or len(set(periods)) != len(periods):
            raise ValueError("period identifiers must be nonempty and unique")
        result = forecast([int(row["completed"]) for row in rows], args.remaining,
                          args.trials, args.max_periods, args.seed)
        result.update(source=str(args.csv_path), period_ids=periods)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
