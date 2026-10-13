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

A = np.ones((2, 3))          # only the shapes matter here
v = np.ones(3)
stack = np.ones((4, 2, 3))

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

#### `np.matmul`

```text
np.matmul(x1, x2, /, out=None, *, axes=<no value>, axis=<no value>, keepdims=False, casting='same_kind', order='K', dtype=None, subok=True, signature=None)
```

The ufunc computes the matrix product. The operator `@` calls it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x1`, `x2` | required | the two inputs. The last axis of `x1` must match the second-to-last axis of `x2`. Leading axes broadcast | a mismatch raises `ValueError`. The `*` operator gives the right shape and the wrong values for two square matrices |
| `out` | `None` | an array to write the result into | an `int64` `out` for `float64` inputs raises `UFuncTypeError` |
| `axes` | `<no value>` | a list of three pairs: the two axes that hold the matrix in `x1`, in `x2` and in the result. With `<no value>`, the last two axes | `axes=[(1, 2), (0, 1), (1, 2)]` on shapes `(3, 2, 4)` and `(4, 5)` gives `(3, 2, 5)`. A pair that names the wrong axes raises `ValueError` |
| `axis` | `<no value>` | one axis for all operands, for a generalized ufunc with one shared core axis | `np.matmul` has more than one core axis, so any value raises `TypeError` |
| `keepdims` | `False` | for a generalized ufunc, keep the reduced core axes with length 1 | `np.matmul` does not support it. `keepdims=True` raises `TypeError` |
| `casting` | `'same_kind'` | which casts are allowed to write the result. As for `np.add` in chapter 6 | — |
| `dtype` | `None` | the type of the computation and of the result | — |
| `order` | `'K'` | the memory order of the result | — |
| `subok` | `True` | with `True`, a subclass of `ndarray` gives a result of that subclass | — |
| `signature` | `None` | the types of the inner loop, as for `np.add` in chapter 6 | — |

Use `@` and `np.dot` only on arrays. Do not use the `*` operator for a matrix product. It multiplies element by element, and for two square matrices it gives a result with the right shape and the wrong values.

## Solve, do not invert

To solve `A x = b` you can compute `inv(A) @ b` or call `solve(A, b)`. Both give the same answer in a textbook. Test them on a Hilbert matrix, which is a standard badly conditioned matrix. The right answer is a vector of ones, because the code builds `b` from it.

```python
import numpy as np


def hilbert(n: int) -> np.ndarray:
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

#### `np.linalg.solve`

```text
np.linalg.solve(a, b)
```

The function solves `a @ x = b` for `x` by a factorization of `a`. It does not form the inverse.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | a square matrix, or a stack of square matrices | a singular matrix raises `LinAlgError: Singular matrix`. A nearly singular matrix gives no error and a wrong answer |
| `b` | required | the right side, with shape `(n,)` or `(n, k)` | the residual is small even when `x` is wrong, so a small residual does not prove a small error |

#### `np.linalg.cond`

```text
np.linalg.cond(x, p=None)
```

The function returns the condition number of a matrix: the factor by which a small change in the input can grow in the answer.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the matrix | for a singular matrix the result is a very large number or `inf` |
| `p` | `None` | the norm. With `None`, the 2-norm, from the singular values | other norms cost less and give other values |

Read the table in three parts.

The condition number `cond(A)` measures how much the answer can change when the input changes a little. A rule of thumb is that the error of the answer is about `cond(A)` times the rounding unit of `float64`, which is 1.1e-16. For `n = 8`, that is 1.5e10 times 1.1e-16, or about 2e-6. The table shows 1.2e-7 for `solve`, and 1.9e-6 for `inv`. The rule gives the order of magnitude.

For `n = 12` the condition number is 1.8e16, which is more than the number of digits that `float64` has. No method can give a good answer. Both are wrong, and the errors are 0.7 and 21.

The last two columns show the real difference. The residual `H @ x - b` of `solve` is at the rounding level for every `n`, even when `x` is wrong. The residual of `inv` is 6e7 times larger at `n = 8` and 1e14 times larger at `n = 12`. The inverse method gives an answer that does not even satisfy the equation. Use the residual as a check: it tells you if the method did its job, where the error tells you only that the problem was hard.

Speed points the same way. The next block times both on a 1000-by-1000 system. It pins the linear algebra library to one thread before it imports NumPy. It also uses `np.random.default_rng` to make a random matrix. Chapter 7 explains that function in the section on random numbers below. A shared machine then gives the same ratio from run to run.

```python
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import timeit

import numpy as np

rng = np.random.default_rng(0)
A, b = rng.random((1000, 1000)), rng.random(1000)

t_solve = min(timeit.repeat(lambda: np.linalg.solve(A, b), number=1, repeat=5))
t_inv = min(timeit.repeat(lambda: np.linalg.inv(A) @ b, number=1, repeat=5))
assert t_inv / t_solve > 1.5
print(f"solve        {t_solve:.3f} s")
print(f"inv(A) @ b   {t_inv:.3f} s")
print(f"ratio        {t_inv / t_solve:.1f}x")
```

```text
# output varies
solve        0.038 s
inv(A) @ b   0.133 s
ratio        3.5x
```

The inverse was slower and gave a worse answer. The assert requires a ratio above 1.5 with one thread.

## Fit with `lstsq`, not with the normal equations

A least-squares fit minimizes `|X β − y|²`. The textbook solution is the normal equations, `(XᵀX) β = Xᵀy`. The problem is that `XᵀX` has the square of the condition number of `X`. Compare the two on polynomial fits, where `y` is built from known coefficients of 1.

```python
import numpy as np

x = np.linspace(0, 1, 50)
print(f"{'degree':>6} {'cond(X)':>9} {'cond(XtX)':>10} {'lstsq error':>12} {'normal-eq error':>16}")
for degree in (3, 6, 9, 12):
    X = np.vander(x, degree + 1)
    y = X @ np.ones(degree + 1)
    via_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    via_normal = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"{degree:>6} {np.linalg.cond(X):9.1e} {np.linalg.cond(X.T @ X):10.1e} {np.abs(via_lstsq - 1).max():12.1e} "
          f"{np.abs(via_normal - 1).max():16.1e}")
```

```text
degree   cond(X)  cond(XtX)  lstsq error  normal-eq error
     3   1.2e+02    1.4e+04      7.1e-15          1.4e-12
     6   2.0e+04    3.9e+08      8.1e-13          4.1e-08
     9   3.6e+06    1.3e+13      5.4e-11          1.3e-03
    12   6.8e+08    4.7e+17      9.7e-09          3.9e-01
```

#### `np.linalg.lstsq`

```text
np.linalg.lstsq(a, b, rcond=None)
```

The function returns the vector `beta` that minimizes the sum of squares of `a @ beta - b`. It returns a tuple of four values: the solution, the residuals, the rank and the singular values.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the design matrix, with one row for each observation | two equal columns give a rank below the number of columns, and the coefficients are not unique |
| `b` | required | the observations, with shape `(m,)` or `(m, k)` | the residuals are empty when the rank is below the number of columns |
| `rcond` | `None` | singular values below `rcond` times the largest are treated as zero. With `None`, the machine precision times the larger side | a larger `rcond` drops small directions and can change the fit |

At degree 9 the error of the normal equations is 1.3e-3 and the error of `lstsq` is 5.4e-11. The normal equations lose 7 digits more. At degree 12 the error is 0.39 where `lstsq` has 1e-8.

The likely cause is the squared condition number. The column `cond(XtX)` shows that `XᵀX` has about the square of the condition number of `X`: 6.8e8 becomes 4.7e17 at degree 12. The inverse of the `float64` epsilon is 4.5e15, so that value is past the limit of `float64`. The function `lstsq` uses a factorization of `X` that does not form `XᵀX`.

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

#### `np.random.default_rng`

```text
np.random.default_rng(seed=None)
```

The function returns a `Generator`, the object that draws random numbers.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `seed` | `None` | an integer, a list of integers, a `SeedSequence` or a generator. With `None`, fresh entropy from the operating system | with `None`, two runs give different numbers, and you cannot repeat the run |

#### `np.random.Generator.random`

```text
np.random.Generator.random(self, size=None, dtype=<class 'numpy.float64'>, out=None)
```

The method returns floats from the half-open range from 0.0 up to, and not including, 1.0.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `size` | `None` | the shape of the result. With `None`, one number | a bare integer gives a 1-D array, and a tuple gives that shape |
| `dtype` | `<class 'numpy.float64'>` | `float64` or `float32` | — |
| `out` | `None` | an array to fill | — |

#### `np.random.Generator.integers`

```text
np.random.Generator.integers(self, low, high=None, size=None, dtype=<class 'numpy.int64'>, endpoint=False)
```

The method returns random integers from the half-open range `[low, high)`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `low` | required | the smallest value. With `high` unset, it is the upper limit and the smallest value is 0 | — |
| `high` | `None` | the upper limit. It is not included | `integers(0, 3)` never gives 3 |
| `size` | `None` | the shape of the result | — |
| `dtype` | `<class 'numpy.int64'>` | the integer type of the result | a `high` that the type cannot hold raises `ValueError`: `integers(0, 300, dtype=np.int8)`. A float type raises `TypeError` |
| `endpoint` | `False` | with `True`, `high` is included | `integers(0, 3, endpoint=True)` can give 3. Without `endpoint`, the largest value is 2 |

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

#### `np.random.SeedSequence`

```text
np.random.SeedSequence(entropy=None, *, spawn_key=(), pool_size=4, n_children_spawned=0)
```

The class turns a seed into the start states of one or more generators. It also makes child sequences.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `entropy` | `None` | the seed: an integer or a list of integers. With `None`, fresh entropy | with `None`, the sequence cannot be repeated |
| `spawn_key` | `()` | the position of this sequence in a tree of sequences | a child made by `spawn` has a key such as `(0,)`, and two children must not share a key |
| `pool_size` | `4` | the size of the internal pool of entropy, in 32-bit words. Keep the default | a value below 4 raises `ValueError` |
| `n_children_spawned` | `0` | the number of children that `spawn` made already. The next child gets this number as its key | a sequence that you build again from the same seed starts at 0, and its children repeat the first children: keys `(0,)` and `(1,)` again. Pass the count to continue at `(2,)` |

#### `np.random.SeedSequence.spawn`

```text
np.random.SeedSequence.spawn(self, n_children)
```

The method returns a list of child sequences. Each one has a different `spawn_key`, and so a different stream.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `n_children` | required | how many children to make | calling it again on the same parent continues the count, so the next children are new ones, not the same ones |

Each child sequence is derived from the parent in a way that is designed to give independent streams. The result of worker 2 does not depend on whether worker 1 ran first. This is the property that lets you move from a loop to a pool of processes and keep the numbers.

## What the documentation promises, and what it does not

The numbers above are stable on one version of NumPy. The NumPy policy is that the streams of `Generator` methods are not guaranteed to be the same between versions. A seed makes a run reproducible on one installation. It does not make it reproducible for ever.

If a published number depends on a stream, write the NumPy version next to the seed. This book does so. NumPy 2.5.4 produced every output.

## The function you build

A line fit gives one slope. To say how well the data fix the slope, resample the rows with replacement, fit again, and look at the spread of the slopes. This is the bootstrap. Doing 1000 fits is a loop in Python, and a few lines of broadcasting here.

```python
import numpy as np


def bootstrap_slopes(x: np.ndarray, y: np.ndarray, n_boot: int, seed) -> np.ndarray:
    """Slope of the least-squares line for `n_boot` resamples of the rows, all at once."""
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (n_boot, len(x)))
    xs, ys = x[idx], y[idx]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    return (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1)


def fit_line(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed=0) -> tuple[float, float, tuple[float, float]]:
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

#### `np.percentile`

```text
np.percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *, weights=None)
```

The function returns the value below which a given percent of the data lie.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the data | — |
| `q` | required | the percent, from 0 to 100, as a number or a list | a fraction such as `0.025` means 0.025 percent. Use `2.5` |
| `axis` | `None` | the axis to work along. With `None`, the flattened array | — |
| `out` | `None` | an array to write the result into. Its shape must be the shape of the result | — |
| `overwrite_input` | `False` | with `True`, the function can sort the input in place to save memory | the input changes: `[5, 1, 4, 2, 3]` became `[1, 2, 3, 4, 5]` after one call for the median |
| `method` | `'linear'` | how to estimate between data points. There are many | `'nearest'` gives data values only, and `'linear'` can give a value that is not in the data |
| `keepdims` | `False` | with `True`, the reduced axes stay with length 1 | with a list for `q`, the result gets a new first axis: `q=[25, 75]` with `axis=1` on a `(3, 4)` array gives `(2, 3, 1)` |
| `weights` | `None` | a weight for each value of `a` | only `method='inverted_cdf'` accepts weights. With the default `'linear'`, it raises `ValueError` |

#### `hilbert`

```text
hilbert(n: int) -> numpy.ndarray
```

The function returns the `n`-by-`n` Hilbert matrix, with the entry `1 / (i + j + 1)`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `n` | required | the size of the matrix | the condition number is 1.6e4 at 4, 1.5e10 at 8 and 1.8e16 at 12 |

#### `bootstrap_slopes`

```text
bootstrap_slopes(x: numpy.ndarray, y: numpy.ndarray, n_boot: int, seed) -> numpy.ndarray
```

The function returns the slope of the least-squares line for each of `n_boot` resamples of the rows.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the 1-D array of predictors | a constant `x` gives `nan` slopes and the warning `invalid value encountered in divide` |
| `y` | required | the 1-D array of observations, with the same length as `x` | a different length raises `IndexError` |
| `n_boot` | required | the number of resamples | the memory is `n_boot * len(x)` values for each of `x[idx]` and `y[idx]` |
| `seed` | required | any seed that `default_rng` accepts | the same seed gives the same slopes, and a different seed gives a different interval |

#### `fit_line`

```text
fit_line(x: numpy.ndarray, y: numpy.ndarray, n_boot: int = 1000, seed=0) -> tuple[float, float, tuple[float, float]]
```

The function returns the intercept, the slope and a 95 percent bootstrap interval for the slope.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the 1-D array of predictors | — |
| `y` | required | the 1-D array of observations | — |
| `n_boot` | `1000` | the number of resamples | with 500 resamples, the measured coverage was 289 of 300 trials |
| `seed` | `0` | the seed of the resampling | the default makes every call give the same interval, which hides the spread between seeds |

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

## How it works

**Solve.** The function `solve` does not compute an inverse. It factors the matrix into a lower and an upper triangle with row swaps, which is Gaussian elimination with partial pivoting. It then solves the two triangular systems. The next block writes this in a few lines and compares it with `np.linalg.solve`.

```python
import numpy as np

from examples.ch07_linalg_random import hilbert


def lu_solve(A, b):
    """Gaussian elimination with partial pivoting, then back substitution."""
    A, b = A.astype(float).copy(), b.astype(float).copy()
    n = len(b)
    for k in range(n):
        p = k + np.argmax(np.abs(A[k:, k]))          # pivot: the largest value in the column
        if p != k:
            A[[k, p]], b[[k, p]] = A[[p, k]], b[[p, k]]
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


rng = np.random.default_rng(0)
A, b = rng.random((50, 50)), rng.random(50)
mine, theirs = lu_solve(A, b), np.linalg.solve(A, b)
print("random 50 x 50, largest difference:", f"{np.abs(mine - theirs).max():.1e}")
print("random 50 x 50, my residual:       ", f"{np.abs(A @ mine - b).max():.1e}")

H = hilbert(8)
true = np.ones(8)
print("Hilbert 8, my error:     ", f"{np.abs(lu_solve(H, H @ true) - true).max():.1e}")
print("Hilbert 8, solve error:  ", f"{np.abs(np.linalg.solve(H, H @ true) - true).max():.1e}")
```

```text
random 50 x 50, largest difference: 1.2e-14
random 50 x 50, my residual:        2.7e-15
Hilbert 8, my error:      7.0e-08
Hilbert 8, solve error:   1.2e-07
```

The hand-written version agrees with `solve` to rounding. On the Hilbert matrix both lose about the same number of digits: the errors are 7.0e-8 and 1.2e-7. The loss is probably the condition number at work and not a fault of the method. The two methods lose the same digits, which supports this, and this book did not isolate the cause. The inverse costs more because it solves for `n` right sides, one for each column of the identity, where `solve` has one.

**Random numbers.** A `Generator` holds a state, and each draw moves the state forward. The same seed gives the same state, so the same stream. A draw of three numbers and then a draw of three more equals one draw of six. The stream does not depend on how you cut it.

```python
import numpy as np

g = np.random.default_rng(5)
state = g.bit_generator.state
print("generator:", state["bit_generator"])
print("state keys:", list(state))

one = np.random.default_rng(9)
two = np.random.default_rng(9)
cut = np.concatenate([one.random(3), one.random(3)])
whole = two.random(6)
print("3 + 3 draws equal 6 draws:", np.array_equal(cut, whole))

children = np.random.SeedSequence(7).spawn(3)
keys = [c.spawn_key for c in children]
starts = {str(np.random.default_rng(c).bit_generator.state["state"]["state"]) for c in children}
print("spawn keys:", keys)
print("different start states:", len(starts), "of", len(children))

print("no seed gives different runs:", np.random.default_rng().random() != np.random.default_rng().random())
```

```text
generator: PCG64
state keys: ['bit_generator', 'state', 'has_uint32', 'uinteger']
3 + 3 draws equal 6 draws: True
spawn keys: [(0,), (1,), (2,)]
different start states: 3 of 3
no seed gives different runs: True
```

Each child has its own key, and the key goes into the start state. This is why children do not repeat each other, and why the result for child 2 does not depend on child 1.

**The invariant.** The same seed and the same sequence of calls give the same numbers on one installation. A solve has a residual at the rounding level, whatever the condition number.

**The failure.** A solve with a large condition number has a small residual and a large error. A seed that you do not set gives a run that you cannot repeat.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.matmul` | The matrix product | matrix product, with broadcast over leading axes |
| `np.linalg.solve` | Solve, do not invert | solution of a square system by factorization |
| `np.linalg.cond` | Solve, do not invert | condition number of a matrix |
| `np.linalg.lstsq` | Fit with `lstsq`, not with the normal equations | least-squares solution, rank and singular values |
| `np.random.default_rng` | Random numbers: create a generator and pass it | a `Generator` from a seed |
| `np.random.Generator.random` | Random numbers: create a generator and pass it | floats from 0.0 up to, not including, 1.0 |
| `np.random.Generator.integers` | Random numbers: create a generator and pass it | integers from `[low, high)` |
| `np.random.SeedSequence` | Parallel streams: `spawn`, not `seed + i` | seed to start states, and child sequences |
| `np.random.SeedSequence.spawn` | Parallel streams: `spawn`, not `seed + i` | independent child sequences |
| `np.percentile` | The function you build | value below which a percent of the data lie |
| `hilbert` | Solve, do not invert | the Hilbert test matrix |
| `bootstrap_slopes` | The function you build | slopes of many row resamples at once |
| `fit_line` | The function you build | line fit with a bootstrap interval |

## What you have now

You have `fit_line`, which fits a line and reports an interval, and which gives the same numbers for the same seed. You have three rules, each with a measurement behind it. `solve` has a residual at the rounding level when `inv` does not. `lstsq` keeps 7 digits that the normal equations lose at degree 9. A spawned seed sequence gives results that do not depend on the order of the workers.

Chapter 8 puts the whole book to work on one program. It ends with a list of the traps in the order that you meet them.
