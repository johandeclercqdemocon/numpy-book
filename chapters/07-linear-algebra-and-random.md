# Chapter 7 — Linear algebra and random numbers

> **Build** — `fit_line(x, y)`, a least-squares line with a bootstrap interval for the slope, which gives the same answer every time that you give it the same seed.
>
> **Theory kept** — solve a system with `solve` and not with the inverse. Fit with `lstsq` and not with the normal equations. Make random numbers from a `Generator` that you create and pass, and not from a global state.
>
> **Source** — New. The conditioning rule is standard numerical linear algebra. The random-number policy is from the NumPy documentation. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch07_linalg_random`, which is [`examples/ch07_linalg_random.py`](../examples/ch07_linalg_random.py).

Two jobs account for most of the NumPy code outside data cleaning. One is to solve for unknown numbers, such as the coefficients of a fit. The other is to draw random numbers, for a simulation, a split or a resample. Both have a textbook form that works on small examples and fails on real ones. This chapter measures how each one fails.

## The matrix product

The operator `@` is the matrix product. It works on vectors, matrices and stacks of matrices.

```python
import numpy as np

A = np.arange(6.0).reshape(2, 3)
v = np.array([1.0, 0.0, -1.0])
stack = np.arange(24.0).reshape(4, 2, 3)

print("matrix @ vector      :", (A @ v).shape)
print("stack of 4 @ vector  :", (stack @ v).shape)
print("stack of 4 @ A.T     :", (stack @ A.T).shape)
```

```text
matrix @ vector      : (2,)
stack of 4 @ vector  : (4, 2)
stack of 4 @ A.T     : (4, 2, 2)
```

The last two lines show the rule. Leading axes act as a stack, and `@` multiplies the last two axes of each matrix. The leading axes broadcast, as in chapter 5. One call does four matrix products.

Use `@` and `np.dot` only on arrays. Do not use the `*` operator for a matrix product. It multiplies element by element, and for two square matrices it gives a result with the right shape and the wrong values.

## Solve, do not invert

To solve `A x = b` you can compute `inv(A) @ b` or call `solve(A, b)`. Both give the same answer in a textbook. Test them on a Hilbert matrix, which is a standard badly conditioned matrix. The right answer is a vector of ones, because the code builds `b` from it.

```python
import numpy as np


def hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


print(f"{'n':>3} {'cond(A)':>9} {'solve error':>12} {'inv error':>10} {'solve resid':>12} {'inv resid':>10}")
for n in (4, 8, 12):
    H = hilbert(n)
    true = np.ones(n)
    b = H @ true
    x_solve = np.linalg.solve(H, b)
    x_inv = np.linalg.inv(H) @ b
    print(f"{n:>3} {np.linalg.cond(H):9.1e} {np.abs(x_solve - true).max():12.1e} "
          f"{np.abs(x_inv - true).max():10.1e} {np.abs(H @ x_solve - b).max():12.1e} "
          f"{np.abs(H @ x_inv - b).max():10.1e}")
```

```text
  n   cond(A)  solve error  inv error  solve resid  inv resid
  4   1.6e+04      6.6e-14    9.1e-13      0.0e+00    9.5e-14
  8   1.5e+10      1.2e-07    1.9e-06      2.2e-16    1.4e-08
 12   1.8e+16      7.0e-01    2.1e+01      4.4e-16    5.4e-02
```

Read the table in three parts.

The condition number `cond(A)` measures how much the answer can change when the input changes a little. A rule of thumb is that the error of the answer is about `cond(A)` times the rounding unit of `float64`, which is 1.1e-16. For `n = 8`, that is 1.5e10 times 1.1e-16, or about 2e-6. The table shows 1.2e-7 for `solve`, and 1.9e-6 for `inv`. The rule gives the order of magnitude.

For `n = 12` the condition number is 1.8e16, which is more than the number of digits that `float64` has. No method can give a good answer. Both are wrong, and the errors are 0.7 and 21.

The last two columns show the real difference. The residual `H @ x - b` of `solve` is at the rounding level for every `n`, even when `x` is wrong. The residual of `inv` is 6e7 times larger at `n = 8` and 1e14 times larger at `n = 12`. The inverse method gives an answer that does not even satisfy the equation. Use the residual as a check: it tells you if the method did its job, where the error tells you only that the problem was hard.

Speed points the same way. For a 2000-by-2000 system, `solve` took 0.18 s and `inv(A) @ b` took 0.48 s on the build machine. The inverse does more work for a worse answer.

## Fit with `lstsq`, not with the normal equations

A least-squares fit minimizes `|X β − y|²`. The textbook solution is the normal equations, `(XᵀX) β = Xᵀy`. The problem is that `XᵀX` has the square of the condition number of `X`. Compare the two on polynomial fits, where `y` is built from known coefficients of 1.

```python
import numpy as np

x = np.linspace(0, 1, 50)
print(f"{'degree':>6} {'cond(X)':>9} {'lstsq error':>12} {'normal-eq error':>16}")
for degree in (3, 6, 9, 12):
    X = np.vander(x, degree + 1)
    y = X @ np.ones(degree + 1)
    via_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    via_normal = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"{degree:>6} {np.linalg.cond(X):9.1e} {np.abs(via_lstsq - 1).max():12.1e} "
          f"{np.abs(via_normal - 1).max():16.1e}")
```

```text
degree   cond(X)  lstsq error  normal-eq error
     3   1.2e+02      7.1e-15          1.4e-12
     6   2.0e+04      8.1e-13          4.1e-08
     9   3.6e+06      5.4e-11          1.3e-03
    12   6.8e+08      9.7e-09          3.9e-01
```

At degree 9 the error of the normal equations is 1.3e-3 and the error of `lstsq` is 5.4e-11. The normal equations lose 7 digits more. At degree 12 the error is 0.39 where `lstsq` has 1e-8. The cause is the squared condition number: `cond(X)` of 6.8e8 at degree 12 becomes 4.6e17 for `XᵀX`, which is past the limit of `float64`. The function `lstsq` uses a factorization of `X` that does not form `XᵀX`.

The function also returns the rank and the singular values, and that is its second use. If the rank is less than the number of columns, two of your columns carry the same information, and the coefficients are not unique. Check the rank of any design matrix that you did not build yourself.

## Random numbers: create a generator and pass it

NumPy has two random-number interfaces. The old one is the global `np.random.seed` and `np.random.rand`. The current one is `np.random.default_rng`, which returns a `Generator` object. Use the current one.

```python
import numpy as np

a = np.random.default_rng(42).random(3)
b = np.random.default_rng(42).random(3)
c = np.random.default_rng(43).random(3)
print("seed 42:", a)
print("seed 42:", b)
print("seed 43:", c)
print("same seed, same numbers:", np.array_equal(a, b))
```

```text
seed 42: [0.77395605 0.43887844 0.85859792]
seed 42: [0.77395605 0.43887844 0.85859792]
seed 43: [0.65229926 0.04377532 0.02002959]
same seed, same numbers: True
```

The reason to avoid the global state is not style. Every function in the process uses the same global state. If one library function draws a number between your two draws, your sequence changes, and the cause is far from the symptom. A `Generator` that you create and pass to the functions that need it has no such coupling. The function signature shows that the function is random, and the caller controls the seed.

## Parallel streams: `spawn`, not `seed + i`

A job with several workers needs one stream per worker, and the streams must not overlap or repeat. The tool is `SeedSequence.spawn`.

```python
import numpy as np


def work(seed_seq):
    return np.random.default_rng(seed_seq).normal(size=1000).mean()


children = np.random.SeedSequence(7).spawn(4)
forward = [work(s) for s in children]
backward = [work(s) for s in reversed(np.random.SeedSequence(7).spawn(4))][::-1]

print("four worker means:", [round(float(m), 4) for m in forward])
print("same in any order:", forward == backward)
print("different streams:", len(set(forward)) == 4)
```

```text
four worker means: [-0.0682, 0.0625, 0.0279, -0.0366]
same in any order: True
different streams: True
```

Each child sequence is derived from the parent in a way that is designed to give independent streams. The result of worker 2 does not depend on whether worker 1 ran first. This is the property that lets you move from a loop to a pool of processes and keep the numbers.

## What the documentation promises, and what it does not

The numbers above are stable on one version of NumPy. The NumPy policy is that the streams of `Generator` methods are not guaranteed to be the same between versions. A seed makes a run reproducible on one installation. It does not make it reproducible for ever.

If a published number depends on a stream, write the NumPy version next to the seed. This book does so. NumPy 2.5.4 produced every output.

## The function you build

A line fit gives one slope. To say how well the data fix the slope, resample the rows with replacement, fit again, and look at the spread of the slopes. This is the bootstrap. Doing 1000 fits is a loop in Python, and a few lines of broadcasting here.

```python
import numpy as np


def bootstrap_slopes(x, y, n_boot, seed):
    """Slope of the least-squares line for `n_boot` resamples of the rows, all at once."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (n_boot, len(x)))
    xs, ys = x[idx], y[idx]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    return (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1)


def fit_line(x, y, n_boot=1000, seed=0):
    """Return (intercept, slope, (low, high)) with a 95% bootstrap interval for the slope."""
    X = np.column_stack([np.ones_like(x), x])
    (b0, b1), *_ = np.linalg.lstsq(X, y, rcond=None)
    low, high = np.percentile(bootstrap_slopes(x, y, n_boot, seed), [2.5, 97.5])
    return float(b0), float(b1), (float(low), float(high))


x = np.linspace(0, 1, 40)
y = 1 + 2 * x + np.random.default_rng(5).normal(0, 0.1, 40)

b0, b1, (lo, hi) = fit_line(x, y, seed=3)
print(f"intercept {b0:.3f}, slope {b1:.3f}, 95% interval for slope ({lo:.3f}, {hi:.3f})")
print("same seed, same interval:     ", fit_line(x, y, seed=3) == fit_line(x, y, seed=3))
print("different seed, same interval:", fit_line(x, y, seed=4)[2] == fit_line(x, y, seed=3)[2])
print("true slope inside interval:   ", lo <= 2.0 <= hi)
```

```text
intercept 1.001, slope 1.934, 95% interval for slope (1.850, 2.014)
same seed, same interval:      True
different seed, same interval: False
true slope inside interval:    True
```

The bootstrap function does three steps from earlier chapters. It draws a `(1000, 40)` array of row numbers. It uses an integer index list, from chapter 4, to build `x[idx]` of shape `(1000, 40)`. It uses `keepdims=True`, from chapter 5, to center each row. No loop appears.

Two checks that the module runs are worth stating. First, each bootstrap slope equals the slope that `lstsq` gives for the same resample, so the closed-form shortcut is right. Second, a bootstrap interval is a claim about coverage, and you can measure the claim. Fit 300 data sets that have a true slope of 2, build the interval for each, and count how often the interval holds 2.

Run this block from the root of the book, so that the import finds the module.

```python
import numpy as np

from examples.ch07_linalg_random import fit_line

x = np.linspace(0, 1, 40)
trials, covered = 300, 0
for t in range(trials):
    y = 1 + 2 * x + np.random.default_rng(1000 + t).normal(0, 0.1, 40)
    _, _, (lo, hi) = fit_line(x, y, n_boot=500, seed=t)
    covered += lo <= 2.0 <= hi
print(f"covered {covered} of {trials}: {covered / trials:.3f}")
```

```text
covered 289 of 300: 0.963
```

The interval is nominally 95 percent. For 300 trials, the standard error of a proportion near 0.95 is 0.013. A result between 0.92 and 0.98 agrees with the claim. The measured coverage is in that range.

## What you have now

You have `fit_line`, which fits a line and reports an interval, and which gives the same numbers for the same seed. You have three rules, each with a measurement behind it. `solve` has a residual at the rounding level when `inv` does not. `lstsq` keeps 7 digits that the normal equations lose at degree 9. A spawned seed sequence gives results that do not depend on the order of the workers.

Chapter 8 puts the whole book to work on one program. It ends with a list of the traps in the order that you meet them.
