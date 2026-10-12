"""Chapter 6: ufunc methods, np.vectorize, and a moving average three ways."""
import timeit

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ma_loop(x, w):
    out = np.empty(len(x) - w + 1)
    for i in range(len(out)):
        out[i] = x[i:i + w].mean()
    return out


def ma_cumsum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def ma_cumsum_centered(x, w):
    """The cumsum version after removing the mean, so the running total stays small."""
    m = x.mean()
    return ma_cumsum(x - m, w) + m


def ma_window(x, w):
    return sliding_window_view(x, w).mean(axis=-1)


def best(f, repeat=3):
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main() -> None:
    rng = np.random.default_rng(2)
    s = rng.random(1_000_000)
    w = 50
    ref = ma_loop(s[:20000], w)
    for f in (ma_cumsum, ma_cumsum_centered, ma_window):
        assert np.allclose(f(s[:20000], w), ref)
    x32 = (s * 100 + 1e4).astype(np.float32)
    truth = ma_window(x32.astype(np.float64), w)
    e_plain = np.abs(ma_cumsum(x32, w) - truth).max()
    e_cent = np.abs(ma_cumsum_centered(x32, w) - truth).max()
    e_win = np.abs(ma_window(x32, w) - truth).max()
    print(f"float32, values from 10000 to 10100: cumsum {e_plain:.3g}, centered {e_cent:.3g}, window {e_win:.3g}")
    assert e_plain > 10 * e_win


if __name__ == "__main__":
    main()
