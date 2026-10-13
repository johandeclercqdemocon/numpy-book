"""Chapter 1: what an ndarray is, and how much faster it is than a list."""
import sys
import timeit

import numpy as np


def describe(a: np.ndarray) -> str:
    """One line that tells the whole layout of an array."""
    return (f"shape={a.shape} dtype={a.dtype} itemsize={a.itemsize} "
            f"strides={a.strides} nbytes={a.nbytes} c_contiguous={a.flags.c_contiguous}")


def main() -> None:
    a = np.array([[3, 1, 4],
                  [1, 5, 9]])
    print(describe(a))
    print(describe(a.T))

    n = 1_000_000
    as_list = list(range(n))
    as_array = np.arange(n)
    t_list = min(timeit.repeat(lambda: sum(as_list), number=5, repeat=5)) / 5
    t_array = min(timeit.repeat(lambda: as_array.sum(), number=5, repeat=5)) / 5
    ratio = t_list / t_array
    print(f"list sum   {t_list * 1e3:8.3f} ms")
    print(f"array sum  {t_array * 1e3:8.3f} ms")
    print(f"ratio      {ratio:8.1f}x")
    assert ratio > 5, "the claim in chapter 1 is that the array sum is at least 5 times faster"
    assert sum(as_list) == int(as_array.sum())

    print(f"list of {n} ints: {sys.getsizeof(as_list) / 1e6:.1f} MB of pointers, "
          f"array: {as_array.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
