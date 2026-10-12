"""SplitMix64 RNG: identical stream in Python and C++ (parity + reproducibility)."""
from __future__ import annotations

_MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))
