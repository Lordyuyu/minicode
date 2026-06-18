from __future__ import annotations

import numpy as np
from typing import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def normalize_vector(vec: Sequence[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return [0.0] * len(vec)
    return (arr / norm).tolist()
