"""One prefill and one batched decode per tick; completed slots are reusable."""

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from miniserve.model import KVCache, Transformer


@dataclass
class Job:
    prompt: list[int]
    max_tokens: int
    submitted: float = field(default_factory=time.perf_counter)
    output: list[int] = field(default_factory=list)
    buffer: deque[int] = field(default_factory=deque)
    times: list[float] = field(default_factory=list)
    slot: int | None = None
    cancelled: bool = False
    finished: bool = False


class Scheduler:
    """Single-owner, greedy CPU scheduler. Consumers drain each job's byte buffer."""

    def __init__(
        self, model: Transformer, *, slots: int = 8, max_pending: int = 128, buffer_limit: int = 32
    ) -> None:
        if min(slots, max_pending, buffer_limit) <= 0:
            raise ValueError("Scheduler capacities must be positive")
        self.model = model.eval()
        self.cache = KVCache(model.config, slots, model.tokens.weight.dtype)
        self.free = deque(range(slots))
        self.pending: deque[Job] = deque()
        self.active: list[Job] = []
        self.max_pending = max_pending
        self.buffer_limit = buffer_limit
        self.completed = 0
        self.cancelled = 0

    def submit(self, prompt: str, max_tokens: int) -> Job:
        tokens = list(prompt.encode())
        if not tokens or max_tokens <= 0:
            raise ValueError("A nonempty prompt and positive token count are required")
        if len(tokens) + max_tokens > self.model.config.context_length:
            raise ValueError("Prompt bytes plus output tokens exceed the context window")
        if self.model.config.vocab_size != 256:
            raise ValueError("The scheduler requires a byte-token model")
        if len(self.pending) >= self.max_pending:
            raise OverflowError("Pending request capacity reached")
        job = Job(tokens, max_tokens)
        self.pending.append(job)
        return job

    def _finish(self, job: Job) -> None:
        if job.slot is not None:
            self.cache.release(job.slot)
            self.free.append(job.slot)
            job.slot = None
        job.finished = True
        if job.cancelled:
            self.cancelled += 1
        else:
            self.completed += 1

    def _emit(self, job: Job, token: int) -> None:
        job.output.append(token)
        job.buffer.append(token)
        job.times.append(time.perf_counter())
        if len(job.output) == job.max_tokens:
            self._finish(job)

    @torch.inference_mode()
    def step(self) -> None:
        for job in self.active:
            if job.cancelled and not job.finished:
                self._finish(job)
        self.active = [job for job in self.active if not job.finished]
        # Capture existing decode work before admitting a new prefill.
        decoding = [job for job in self.active if len(job.buffer) < self.buffer_limit]
        remaining = deque()
        for job in self.pending:
            if job.cancelled:
                self._finish(job)
            else:
                remaining.append(job)
        self.pending = remaining
        if self.free and self.pending:
            job = self.pending.popleft()
            job.slot = self.free.popleft()
            logits = self.model(torch.tensor([job.prompt]), self.cache, torch.tensor([job.slot]))
            self._emit(job, int(logits[0, -1].argmax().item()))
            if not job.finished:
                self.active.append(job)
        if decoding:
            tokens = torch.tensor([[job.output[-1]] for job in decoding])
            slots = torch.tensor([job.slot for job in decoding])
            logits = self.model(tokens, self.cache, slots)
            for job, token in zip(decoding, logits[:, -1].argmax(-1).tolist(), strict=True):
                self._emit(job, token)
        self.active = [job for job in self.active if not job.finished]

    @property
    def busy(self) -> bool:
        return bool(self.pending or self.active)
