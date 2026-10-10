"""The TinyTrain decoder, extended with explicit cache writes and position masks."""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context_length, self.width, self.heads, self.layers) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Width must divide evenly across heads")


class KVCache:
    def __init__(self, config: ModelConfig, slots: int, dtype: torch.dtype) -> None:
        if slots <= 0:
            raise ValueError("Cache slots must be positive")
        shape = (
            config.layers,
            slots,
            config.heads,
            config.context_length,
            config.width // config.heads,
        )
        self.keys = torch.empty(shape, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.lengths = torch.zeros(slots, dtype=torch.long)

    @property
    def bytes(self) -> int:
        return 2 * self.keys.numel() * self.keys.element_size() + self.lengths.numel() * 8

    def release(self, slot: int) -> None:
        self.lengths[slot] = 0


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: KVCache | None,
        slots: Tensor | None,
        layer: int,
    ) -> Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        q, k, v = [x.view(batch, length, self.heads, -1).transpose(1, 2) for x in (q, k, v)]
        if cache is not None:
            # Advanced indexing is [batch, time, heads, head_width].
            cache.keys[layer, slots[:, None], :, positions] = k.transpose(1, 2)
            cache.values[layer, slots[:, None], :, positions] = v.transpose(1, 2)
            end = int(positions[:, -1].max().item()) + 1
            k = cache.keys[layer, slots, :, :end]
            v = cache.values[layer, slots, :, :end]
        key_positions = torch.arange(k.shape[-2])
        visible = key_positions[None, None, :] <= positions[:, :, None]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        # A decode query must see its entire prefix; a Q=1 causal triangle is insufficient.
        scores = scores.masked_fill(~visible[:, None], float("-inf"))
        probabilities = scores.softmax(-1)
        if cache is not None:
            # Unwritten cache bytes may be NaN; masked probabilities alone cannot erase NaN * 0.
            valid_keys = key_positions[None, :] <= positions[:, -1, None]
            v = v.masked_fill(~valid_keys[:, None, :, None], 0)
        attended = probabilities @ v
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        cache: KVCache | None,
        slots: Tensor | None,
        layer: int,
    ) -> Tensor:
        hidden = hidden + self.attention(
            self.attention_norm(hidden), positions, cache, slots, layer
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, tokens: Tensor, cache: KVCache | None = None, slots: Tensor | None = None
    ) -> Tensor:
        batch, length = tokens.shape
        if not 0 < length <= self.config.context_length:
            raise ValueError("Input length must be within the context window")
        if cache is None:
            starts = torch.zeros(batch, dtype=torch.long)
        else:
            if slots is None or slots.shape != (batch,) or slots.unique().numel() != batch:
                raise ValueError("Cached rows require distinct slots, one per sequence")
            starts = cache.lengths[slots].clone()
        positions = starts[:, None] + torch.arange(length)
        if positions.max() >= self.config.context_length:
            raise ValueError("Cache would exceed the context window")
        hidden = self.tokens(tokens) + self.positions(positions)
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, positions, cache, slots, index)
        if cache is not None:
            cache.lengths[slots] += length
        return F.linear(self.norm(hidden), self.tokens.weight)


def load_model(path: Path | None = None, seed: int = 42) -> Transformer:
    torch.manual_seed(seed)
    if path is None:
        return Transformer(ModelConfig()).eval()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.eval()
