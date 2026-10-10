import pytest
import torch

from miniserve.model import KVCache, ModelConfig, Transformer


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-6), (torch.float64, 1e-12)])
def test_mixed_length_decode_matches_full_prefix_with_poisoned_unused_cache(dtype, atol):
    torch.manual_seed(7)
    model = Transformer(ModelConfig(width=16, layers=2, context_length=32)).to(dtype).eval()
    cache = KVCache(model.config, slots=3, dtype=dtype)
    cache.keys.fill_(float("nan"))
    cache.values.fill_(float("nan"))
    prefixes = [torch.tensor([[5, 7]]), torch.tensor([[9, 3, 2, 11, 4]])]
    slots = torch.tensor([2, 0])
    with torch.inference_mode():
        for prefix, slot in zip(prefixes, slots, strict=True):
            actual = model(prefix, cache, slot.reshape(1))
            torch.testing.assert_close(actual, model(prefix), atol=atol, rtol=atol)
        for step in range(4):
            tokens = torch.tensor([[step + 12], [step + 23]])
            actual = model(tokens, cache, slots)[:, -1]
            for row in range(2):
                prefixes[row] = torch.cat((prefixes[row], tokens[row : row + 1]), dim=1)
                expected = model(prefixes[row])[:, -1]
                torch.testing.assert_close(actual[row : row + 1], expected, atol=atol, rtol=atol)


def test_reused_slot_cannot_read_previous_request():
    torch.manual_seed(9)
    model = Transformer(ModelConfig(width=16, layers=1, context_length=16)).double().eval()
    cache = KVCache(model.config, 1, torch.float64)
    slot = torch.tensor([0])
    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3, 4, 5, 6]]), cache, slot)
        cache.release(0)
        prompt = torch.tensor([[17, 18]])
        torch.testing.assert_close(
            model(prompt, cache, slot), model(prompt), atol=1e-12, rtol=1e-12
        )
    assert cache.lengths.tolist() == [2]


def test_cache_overflow_and_duplicate_slots_are_rejected():
    model = Transformer(ModelConfig(width=16, layers=1, context_length=4)).eval()
    cache = KVCache(model.config, 2, torch.float32)
    with torch.inference_mode():
        model(torch.tensor([[1, 2, 3, 4]]), cache, torch.tensor([0]))
        with pytest.raises(ValueError, match="exceed"):
            model(torch.tensor([[5]]), cache, torch.tensor([0]))
        with pytest.raises(ValueError, match="distinct"):
            model(torch.tensor([[1], [2]]), cache, torch.tensor([1, 1]))
