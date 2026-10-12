"""Chapter 3: views, copies, and the cost of walking memory in the wrong order."""
import timeit

import numpy as np


def root(a: np.ndarray) -> np.ndarray:
    """Follow .base links to the array that owns the memory of `a`."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def best(f, repeat: int = 3) -> float:
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main() -> None:
    a = np.arange(12).reshape(3, 4)
    view, fancy, masked = a[:, 1:3], a[[0, 2]], a[a > 5]
    assert np.shares_memory(a, view)
    assert not np.shares_memory(a, fancy)
    assert not np.shares_memory(a, masked)
    assert root(view) is root(a)
    assert root(fancy) is fancy

    n = 6000
    big = np.random.default_rng(0).random((n, n))
    t_copy = best(lambda: big.copy())
    t_tcopy = best(lambda: np.ascontiguousarray(big.T))
    t_row = best(lambda: [big[j, :].sum() for j in range(0, n, 10)])
    t_col = best(lambda: [big[:, j].sum() for j in range(0, n, 10)])
    print(f"transposed copy / plain copy   {t_tcopy / t_copy:5.1f}x")
    print(f"column loop / row loop         {t_col / t_row:5.1f}x")
    assert t_tcopy / t_copy > 1.3
    assert t_col / t_row > 3


if __name__ == "__main__":
    main()
