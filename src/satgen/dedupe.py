"""Near-duplicate detection: MinHash over word 3-grams, LSH bands to pick candidate pairs, exact Jaccard on those."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

_PRIME = (1 << 61) - 1
_TOKEN = re.compile(r"[a-z0-9]+")


def shingles(text: str, k: int = 3) -> set[str]:
    words = _TOKEN.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


@dataclass
class MinHashIndex:
    num_perm: int = 128
    bands: int = 32
    seed: int = 1
    _params: list[tuple[int, int]] = field(init=False)
    _buckets: list[dict[tuple, list[str]]] = field(init=False)
    _shingles: dict[str, set[str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = _Lcg(self.seed)
        self._params = [(rng.next() % (_PRIME - 1) + 1, rng.next() % _PRIME) for _ in range(self.num_perm)]
        self._buckets = [defaultdict(list) for _ in range(self.bands)]

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands

    def signature(self, sh: set[str]) -> list[int]:
        hashes = [_h64(s) for s in sh] or [0]
        return [min((a * h + b) % _PRIME for h in hashes) for a, b in self._params]

    def _band_keys(self, sig: list[int]):
        r = self.rows
        for i in range(self.bands):
            yield i, tuple(sig[i * r : (i + 1) * r])

    def query(self, text: str) -> list[tuple[str, float]]:
        """Return (key, exact Jaccard) for indexed items that collide with `text`, best first."""
        sh = shingles(text)
        candidates: set[str] = set()
        for i, band in self._band_keys(self.signature(sh)):
            candidates.update(self._buckets[i].get(band, ()))
        scored = [(c, jaccard(sh, self._shingles[c])) for c in candidates]
        return sorted(scored, key=lambda x: -x[1])

    def add(self, key: str, text: str) -> None:
        sh = shingles(text)
        self._shingles[key] = sh
        for i, band in self._band_keys(self.signature(sh)):
            self._buckets[i][band].append(key)

    def __len__(self) -> int:
        return len(self._shingles)


class _Lcg:
    """Tiny deterministic generator so signatures are stable across Python versions."""

    def __init__(self, seed: int):
        self.state = seed or 1

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % (1 << 64)
        return self.state
