import pytest
import torch

from miniserve.model import ModelConfig, Transformer
from miniserve.scheduler import Scheduler


def model():
    torch.manual_seed(42)
    return Transformer(ModelConfig(width=16, context_length=32, layers=1)).eval()


@torch.inference_mode()
def reference(model, prompt, count):
    tokens = torch.tensor([list(prompt.encode())])
    output = []
    for _ in range(count):
        next_token = model(tokens)[:, -1].argmax(-1, keepdim=True)
        output.append(next_token.item())
        tokens = torch.cat((tokens, next_token), dim=1)
    return output


def test_continuous_batching_preserves_greedy_outputs_and_reclaims_slots():
    transformer = model()
    scheduler = Scheduler(transformer, slots=2)
    workload = [("short", 1), ("longer prompt", 8), ("third", 3), ("fourth", 5)]
    jobs = [scheduler.submit(prompt, count) for prompt, count in workload]
    for _ in range(30):
        if not scheduler.busy:
            break
        scheduler.step()
        for job in jobs:
            job.buffer.clear()
    assert not scheduler.busy
    for job, (prompt, count) in zip(jobs, workload, strict=True):
        assert job.output == reference(transformer, prompt, count)
    assert len(scheduler.free) == 2
    assert scheduler.cache.lengths.tolist() == [0, 0]
    assert scheduler.completed == 4


def test_active_and_queued_cancellation_releases_capacity():
    scheduler = Scheduler(model(), slots=1, max_pending=2)
    active = scheduler.submit("first", 8)
    scheduler.step()
    queued = scheduler.submit("second", 5)
    survivor = scheduler.submit("third", 2)
    active.cancelled = queued.cancelled = True
    scheduler.step()
    assert active.finished and queued.finished
    assert queued.output == []
    assert survivor.output
    assert scheduler.cancelled == 2
    scheduler.step()
    assert survivor.finished
    assert len(scheduler.free) == 1


def test_slow_consumer_pauses_without_blocking_other_requests():
    scheduler = Scheduler(model(), slots=2, buffer_limit=2)
    slow = scheduler.submit("slow", 6)
    fast = scheduler.submit("fast", 5)
    for _ in range(10):
        scheduler.step()
        fast.buffer.clear()
    assert fast.finished
    assert len(slow.output) == len(slow.buffer) == 2
    while scheduler.busy:
        slow.buffer.clear()
        scheduler.step()
    assert slow.finished


def test_admission_uses_bytes_and_bounds_pending_requests():
    scheduler = Scheduler(model(), slots=1, max_pending=1)
    with pytest.raises(ValueError, match="context"):
        scheduler.submit("😀" * 8, 1)
    scheduler.submit("one", 2)
    with pytest.raises(OverflowError, match="capacity"):
        scheduler.submit("two", 2)
