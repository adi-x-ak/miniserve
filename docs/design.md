# Cache and scheduling contracts

## Position-correct attention

The decoder architecture and checkpoint field names derive from
[TinyTrain](https://github.com/adi-x-ak/tinytrain), under its MIT license. Learned
positions, tied embeddings, normalization, attention projections, and MLPs are
unchanged. This repository adds its own cache path and serving scheduler.

Each layer stores keys and values in fixed slot tensors. A slot's length advances
once per forward pass, after all layers write at the same positions. Attention
permits a key exactly when its absolute position is at or before the query's
position. A one-token decode query therefore sees its entire prefix. A one-by-many
causal triangle with a top-left alignment would incorrectly restrict it.

Mixed-length decode batches use a prefix up to the batch's maximum key position.
Shorter rows mask unused scores and zero their unused values. Zeroing values
matters because multiplying a masked probability of zero by uninitialized NaN
memory still produces NaN. Tests explicitly fill unwritten cache storage with
NaNs and compare against full-prefix inference in float32 and float64. Float32
comparisons allow `1e-6` absolute/relative error; float64 allows `1e-12`.

Releasing a slot resets its length. Old bytes remain allocated but cannot enter
attention; slot-reuse tests check this isolation. Prompt plus output is bounded
by the configured context. Sliding or truncating learned positions would change
the full-prefix reference and is deliberately unsupported.

## Request lifecycle

The scheduler has one owner. Each tick removes cancellations, admits at most one
prefill if a slot is free, then decodes one token for all ready existing requests
in a batch. A prefill produces the first output token. Subsequent ticks feed back
the last generated token. Finished requests return their slot immediately.

Pending admission is bounded; saturation returns HTTP 429. Prompt/output bounds
return 422. Each token buffer is bounded. A full buffer pauses that request's
decode until its consumer drains it; other ready requests remain batchable. A
slow client can retain a slot until it disconnects, so backpressure does not
guarantee fairness under arbitrarily slow consumers. An idle-client timeout is
not implemented.

The FastAPI lifespan owns the scheduler pump. Its small CPU forwards run on the
event loop, yielding between ticks; this is a deliberate single-process local
design. Larger blocking kernels would need a worker boundary. Streaming cleanup
marks the job cancelled; the next tick reclaims its slot. The HTTP verification
script starts a real server, closes after the first byte event, checks the
cancellation counter, and proves slot reuse with a new completed request.

Generation is greedy. Server and Python API support byte vocabulary 256. Model
weights are random by default for a zero-download demo; a TinyTrain checkpoint
can replace them. No language-quality result is inferred from these workloads.

## Performance experiment

All 32 requests are available at time zero. The workload repeats four prompt/output
length pairs eight times: (8,8), (24,64), (16,16), (32,32) bytes. It produces 960
output bytes using seed-42 weights, width 64, two layers, four heads, context 128.

The serial cached, fixed-group, and continuous paths share the same implementation.
Fixed groups admit the next group only after all jobs in the current group finish.
Both eight-slot policies use the same one-prefill-per-tick rule and stop computing
completed requests. Continuous scheduling refills free slots from the backlog.
Full-prefix serial generation is an additional reference. Every measured run
must reproduce its output bytes exactly.

Each policy warms up once, then runs five times with rotated policy order. The
timer includes request admission and generation but excludes model/cache
allocation. First-token time includes queueing from the common arrival instant.
Per-token intervals are pooled within requests; percentiles use the lower
ordered observation at `floor(p * (n - 1))`. Tables report medians of the five
run-level measurements, not a latency measured across the HTTP transport.

Machine: Apple M4 Pro, 24 GiB RAM, macOS 15.7.5, Python 3.12.14, PyTorch 2.14.1,
one CPU thread. Batch timing depends on hardware and workload. KV byte counts
are exact storage capacity, not peak resident memory. Check the raw JSON for all
repetitions and inter-token latency tradeoffs.

## Prior work

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu): iteration-level scheduling.
- [PyTorch attention API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): causal-mask alignment and attention semantics.
- [Hugging Face cache documentation](https://huggingface.co/docs/transformers/en/cache_explanation): autoregressive key/value reuse.

The engine demonstrates established ideas with small, inspectable code. It does
not claim to introduce a new serving algorithm.
