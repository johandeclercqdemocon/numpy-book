"""Chapter 2: dtypes, silent wraparound and a checked add that refuses to wrap."""
import numpy as np


def checked_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two integer arrays. Raise OverflowError if a result does not fit the dtype of `a`."""
    if a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
        raise TypeError("checked_add takes integer arrays only")
    if a.dtype == np.uint64 or b.dtype == np.uint64:
        raise TypeError("uint64 has no wider integer type to check against")
    wide = a.astype(np.int64) + b.astype(np.int64)
    info = np.iinfo(a.dtype)
    if wide.size and (wide.min() < info.min or wide.max() > info.max):
        bad = int(np.flatnonzero((wide < info.min) | (wide > info.max))[0])
        raise OverflowError(f"element {bad}: {int(wide.flat[bad])} does not fit {a.dtype}")
    return wide.astype(a.dtype)


def main() -> None:
    pixels = np.array([200, 250, 10], dtype=np.uint8)
    boost = np.array([100, 100, 100], dtype=np.uint8)
    wrapped = pixels + boost
    assert wrapped.tolist() == [44, 94, 110]
    try:
        checked_add(pixels, boost)
    except OverflowError as e:
        print("refused:", e)
    else:
        raise AssertionError("checked_add must refuse this input")
    ok = checked_add(pixels, np.array([10, 5, 100], dtype=np.uint8))
    assert ok.tolist() == [210, 255, 110]

    x = np.full(10_000_000, 0.1, dtype=np.float32)
    exact = float(np.float32(0.1)) * x.size
    pairwise = float(x.sum())
    running = float(np.cumsum(x)[-1])
    print(f"pairwise sum error {abs(pairwise - exact) / exact:.2e}")
    print(f"running sum error  {abs(running - exact) / exact:.2e}")
    assert abs(pairwise - exact) / exact < 1e-6
    assert abs(running - exact) / exact > 1e-2


if __name__ == "__main__":
    main()
