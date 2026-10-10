# MiniServe

[![Check](https://github.com/adi-x-ak/miniserve/actions/workflows/check.yml/badge.svg)](https://github.com/adi-x-ak/miniserve/actions/workflows/check.yml)

A readable CPU inference engine: **explicit KV caching, continuous batching,
bounded admission, streaming, and cancellation**. The token-generation loop and
scheduler live here; generation is greedy and byte-level.

## Evidence

- Cached logits match full-prefix logits for different sequence lengths and reused slots.
- Tests poison unwritten cache memory with NaNs to catch invalid reads.
- Continuous scheduling preserves greedy outputs across mixed-length requests.
- A real HTTP disconnect cancels generation and a subsequent request reuses the slot.
- Slow consumers pause their generation while other requests continue.

The measured 32-request workload uses one CPU thread on an M4 Pro, a 120k-scale
randomly initialized Transformer, and five repeated runs. All policies generate
the same 960 output bytes. Median results:

| Policy | Output bytes/s | Median first-token latency | p95 first-token latency | KV storage |
|---|---:|---:|---:|---:|
| Full-prefix, serial | 4,199 | 109.1 ms | 201.6 ms | 0 |
| Cached, serial | 5,161 | 89.9 ms | 164.6 ms | 128 KiB |
| Fixed groups of 8 | 13,536 | 28.5 ms | 55.6 ms | 1 MiB |
| Continuous, 8 slots | **18,975** | **14.0 ms** | **34.2 ms** | 1 MiB |

Continuous scheduling is **1.40×** the fixed-group throughput in this experiment.
Batching increases some inter-token intervals despite improving total throughput
and queueing latency. These are in-process CPU measurements, excluding HTTP;
they make no comparison to GPU serving engines. KV storage excludes model weights,
temporary tensors, and allocator overhead.

See [raw repetitions](results/benchmark.json), [cache evidence](results/cache.json), [HTTP evidence](results/http.json),
and [method and constraints](docs/design.md).

## Run

Python 3.11 or newer. Run commands from the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
miniserve --slots 8 --port 8000
```

In another terminal:

```bash
python examples/client.py --prompt 'Hello' --tokens 32
```

Default weights are random and useful for exercising the engine. For trained
weights, train the small model in [TinyTrain](https://github.com/adi-x-ak/tinytrain)
and pass `miniserve --checkpoint /path/to/checkpoint.pt`. This architecture is
compatible with its checkpoint format. Short training still produces a tiny
text model, not a conversational assistant.

`POST /generate` accepts `{"prompt": "Hello", "max_tokens": 32}` and returns SSE
events containing a byte integer, then `{"done": true}`. Byte events preserve
UTF-8 across token boundaries; the example client performs incremental decoding.
`GET /health` reports capacity use and completion/cancellation counts.

## Read the code

| File | Responsibility |
|---|---|
| [model.py](src/miniserve/model.py) | Decoder, cache storage, writes, position-aware attention |
| [scheduler.py](src/miniserve/scheduler.py) | Admission, prefill, batched decode, backpressure, slot recovery |
| [api.py](src/miniserve/api.py) | Scheduler lifecycle and HTTP streaming |

```python
from miniserve.model import load_model
from miniserve.scheduler import Scheduler

scheduler = Scheduler(load_model(), slots=2)
job = scheduler.submit("Hello", max_tokens=16)
while scheduler.busy:
    scheduler.step()
    job.buffer.clear()  # A consumer must drain its bounded token buffer.
print(bytes(job.output).decode(errors="replace"))
```

## Verify

```bash
ruff check .
ruff format --check .
pytest -q
python scripts/verify_cache.py
python scripts/verify_http.py
python scripts/benchmark.py
```

[requirements-repro.txt](requirements-repro.txt) records the experiment's Python
3.12/macOS packages. Timings and rounding can change across environments.

CPU-only, one process, one architecture. Prompt bytes plus requested output must
fit the context window. There is no sliding-window cache, paged attention,
sampling, model sharding, or production deployment layer. MIT-licensed;
architecture attribution and implementation details are in [design.md](docs/design.md).
