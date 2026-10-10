"""Record cached/full-prefix error for a mixed-length batch in both CPU dtypes."""

import argparse
import json
from pathlib import Path

import torch

from miniserve.model import KVCache, load_model


@torch.inference_mode()
def compare(dtype: torch.dtype) -> float:
    model = load_model().to(dtype)
    cache = KVCache(model.config, 4, dtype)
    cache.keys.fill_(float("nan"))
    cache.values.fill_(float("nan"))
    generator = torch.Generator().manual_seed(7)
    prefixes = [torch.randint(256, (1, length), generator=generator) for length in (6, 19, 27)]
    slots = torch.tensor([3, 0, 2])
    maximum = 0.0
    for prefix, slot in zip(prefixes, slots, strict=True):
        cached = model(prefix, cache, slot.reshape(1))
        if not torch.isfinite(cached).all():
            raise AssertionError("Cached prefill produced nonfinite logits")
        maximum = max(maximum, (cached - model(prefix)).abs().max().item())
    for _ in range(8):
        tokens = torch.randint(256, (3, 1), generator=generator)
        cached = model(tokens, cache, slots)[:, -1]
        if not torch.isfinite(cached).all():
            raise AssertionError("Cached decode produced nonfinite logits")
        for row in range(3):
            prefixes[row] = torch.cat((prefixes[row], tokens[row : row + 1]), dim=1)
            expected = model(prefixes[row])[:, -1]
            maximum = max(maximum, (cached[row] - expected[0]).abs().max().item())
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/cache.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    errors = {"float32": compare(torch.float32), "float64": compare(torch.float64)}
    for dtype, tolerance in (("float32", 1e-6), ("float64", 1e-12)):
        if not errors[dtype] <= tolerance:
            raise AssertionError(f"Cache parity failed for {dtype}: {errors[dtype]}")
    report = {
        "torch": torch.__version__,
        "prompt_lengths": [6, 19, 27],
        "decode_steps": 8,
        "unused_memory": "NaN",
        "max_logit_error": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
