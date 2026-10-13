"""Chapter 5: broadcasting, and three ways to compute all pairwise distances."""
import timeit
import tracemalloc

import numpy as np


def dist_loop(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distances between every row of A and every row of B, with two Python loops."""
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i, j] = np.sqrt(((a - b) ** 2).sum())
    return out


def dist_broadcast(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact. Builds an (n, m, d) temporary array."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast and small, but loses precision when points are close together and far from the origin."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def dist_dot_centered(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The dot version, after moving the origin to the middle of the data."""
    c = np.concatenate([A, B]).mean(axis=0)
    return dist_dot(A - c, B - c)


def peak_mb(f, *args) -> float:
    tracemalloc.start()
    f(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def main() -> None:
    rng = np.random.default_rng(0)
    A, B = rng.random((300, 16)), rng.random((400, 16))
    ref = dist_loop(A, B)
    assert np.allclose(dist_broadcast(A, B), ref)
    assert np.allclose(dist_dot(A, B), ref)

    A2, B2 = rng.random((2000, 64)), rng.random((2000, 64))
    mb_bc, mb_dot = peak_mb(dist_broadcast, A2, B2), peak_mb(dist_dot, A2, B2)
    print(f"peak memory: broadcast {mb_bc:.0f} MB, dot {mb_dot:.0f} MB")
    assert mb_bc > 10 * mb_dot

    P = rng.random((60, 3)) * 0.01 + 1e4
    ref = dist_loop(P, P)
    err_dot = np.abs(dist_dot(P, P) - ref).max()
    err_cen = np.abs(dist_dot_centered(P, P) - ref).max()
    print(f"max error, typical distance {ref.mean():.4f}: dot {err_dot:.1e}, centered {err_cen:.1e}")
    assert err_dot > 100 * err_cen


if __name__ == "__main__":
    main()
