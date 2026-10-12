"""Chapter 7: solve versus inv, lstsq versus the normal equations, and reproducible random streams."""
import numpy as np


def hilbert(n: int) -> np.ndarray:
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


def bootstrap_slopes(x, y, n_boot: int, seed) -> np.ndarray:
    """Slope of the least-squares line for `n_boot` resamples of the rows, all at once."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (n_boot, len(x)))
    xs, ys = x[idx], y[idx]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    return (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1)


def fit_line(x, y, n_boot: int = 1000, seed=0):
    """Return (intercept, slope, (low, high)) with a 95% bootstrap interval for the slope."""
    X = np.column_stack([np.ones_like(x), x])
    (b0, b1), *_ = np.linalg.lstsq(X, y, rcond=None)
    low, high = np.percentile(bootstrap_slopes(x, y, n_boot, seed), [2.5, 97.5])
    return float(b0), float(b1), (float(low), float(high))


def main() -> None:
    # Each bootstrap slope must equal the slope that lstsq gives for the same resample.
    g = np.random.default_rng(5)
    x = np.linspace(0, 1, 40)
    y = 1 + 2 * x + g.normal(0, 0.1, 40)
    slopes = bootstrap_slopes(x, y, 5, seed=11)
    idx = np.random.default_rng(11).integers(0, 40, (5, 40))
    for k in range(5):
        X = np.column_stack([np.ones(40), x[idx[k]]])
        assert np.isclose(np.linalg.lstsq(X, y[idx[k]], rcond=None)[0][1], slopes[k])
    assert fit_line(x, y, seed=3) == fit_line(x, y, seed=3)

    covered, trials = 0, 300
    for t in range(trials):
        yt = 1 + 2 * x + np.random.default_rng(1000 + t).normal(0, 0.1, 40)
        _, _, (lo, hi) = fit_line(x, yt, n_boot=500, seed=t)
        covered += lo <= 2.0 <= hi
    cov = covered / trials
    print(f"95% interval covers the true slope in {cov:.3f} of {trials} trials")
    assert 0.88 < cov <= 1.0


if __name__ == "__main__":
    main()
