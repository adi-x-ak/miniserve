# Code standards

Keep cache math, scheduling decisions, and HTTP behavior explicit. Support CPU,
one decoder architecture, and greedy byte-token generation. Do not add provider
adapters or an external generation engine.

Cache changes require cached/full-prefix parity for mixed lengths and reused
slots. Scheduler changes require cancellation, bounded admission, backpressure,
and output parity checks. Publish actual workload and hardware with performance
claims. A local CPU comparison does not establish GPU serving performance.

Run `ruff check .`, `ruff format --check .`, and `pytest -q`. Keep numerical
evidence and HTTP disconnect verification reproducible.
