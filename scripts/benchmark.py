"""Compare equal-output generation policies on a fixed mixed-length CPU workload."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from miniserve.model import Transformer, load_model
from miniserve.scheduler import Job, Scheduler

WORKLOAD = [("x" * length, count) for length, count in [(8, 8), (24, 64), (16, 16), (32, 32)] * 8]


@torch.inference_mode()
def run(model: Transformer, policy: str) -> tuple[list[Job], dict]:
    slots = 1 if policy in ("uncached", "cached_serial") else 8
    scheduler = None if policy == "uncached" else Scheduler(model, slots=slots)
    started = time.perf_counter()
    jobs = []
    if policy == "uncached":
        for prompt, count in WORKLOAD:
            job = Job(list(prompt.encode()), count, submitted=started)
            tokens = torch.tensor([job.prompt])
            for _ in range(count):
                token = model(tokens)[:, -1].argmax(-1, keepdim=True)
                job.output.append(token.item())
                job.times.append(time.perf_counter())
                tokens = torch.cat((tokens, token), dim=1)
            jobs.append(job)
    else:
        group_size = slots if policy == "static" else len(WORKLOAD)
        for start in range(0, len(WORKLOAD), group_size):
            group = [
                scheduler.submit(prompt, count)
                for prompt, count in WORKLOAD[start : start + group_size]
            ]
            for job in group:
                job.submitted = started
            jobs.extend(group)
            while scheduler.busy:
                scheduler.step()
                for job in group:
                    job.buffer.clear()
    elapsed = time.perf_counter() - started
    first = sorted(job.times[0] - job.submitted for job in jobs)
    intervals = sorted(
        b - a for job in jobs for a, b in zip(job.times, job.times[1:], strict=False)
    )
    metrics = {
        "seconds": elapsed,
        "tokens_per_second": sum(job.max_tokens for job in jobs) / elapsed,
        "ttft_p50_ms": statistics.median(first) * 1000,
        "ttft_p95_ms": first[int(0.95 * (len(first) - 1))] * 1000,
        "token_interval_p50_ms": statistics.median(intervals) * 1000,
        "token_interval_p95_ms": intervals[int(0.95 * (len(intervals) - 1))] * 1000,
        "kv_bytes": 0 if policy == "uncached" else scheduler.cache.bytes,
    }
    return jobs, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("Repeats must be positive")
    torch.set_num_threads(1)
    model = load_model()
    reference, _ = run(model, "uncached")
    expected = [job.output for job in reference]
    policies = ("uncached", "cached_serial", "static", "continuous")
    measurements = {policy: [] for policy in policies}
    for policy in policies:
        run(model, policy)  # Warm up each path; these observations are excluded.
    for repeat in range(args.repeats):
        # Rotate execution order to reduce systematic warm-cache/order effects.
        order = policies[repeat % 4 :] + policies[: repeat % 4]
        for policy in order:
            jobs, metrics = run(model, policy)
            if [job.output for job in jobs] != expected:
                raise AssertionError(f"{policy} changed greedy outputs")
            measurements[policy].append(metrics)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": 1,
        "weights": "seed-42 random TinyTrain architecture; no language-quality claim",
        "slots": 8,
        "arrival": "all requests arrive at time zero",
        "workload": [
            {"prompt_bytes": len(prompt), "output_bytes": count} for prompt, count in WORKLOAD
        ],
        "output_parity": True,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for policy, records in measurements.items():
        print(
            policy,
            {key: round(statistics.median(r[key] for r in records), 3) for key in records[0]},
        )


if __name__ == "__main__":
    main()
