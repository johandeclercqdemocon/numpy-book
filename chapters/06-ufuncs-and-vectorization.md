# Chapter 6 — Ufuncs and vectorization

> **Build** — a moving average in four versions, a loop, a cumulative-sum version, a window version and a convolution, with a measured time and a measured precision for each.
>
> **Theory kept** — a ufunc is a function that runs element by element in compiled code. It has methods (`reduce`, `accumulate`, `outer`) that apply it along an axis. Vectorization is the act of rewriting a loop as calls to ufuncs. It does not exist for every loop.
>
> **Source** — New. The ufunc methods are in the NumPy documentation. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch06_ufuncs`, which is [`examples/ch06_ufuncs.py`](../examples/ch06_ufuncs.py).

A ufunc, short for universal function, is the unit of speed in NumPy. `np.add`, `np.multiply`, `np.sqrt`, `np.maximum` and about 90 others are ufuncs. The operators `+` and `*` call them. A ufunc takes arrays, loops over the elements in compiled code, and returns an array. It does the Python-level work once for the whole array, and not once per element.

This chapter shows three methods that make a ufunc more than an element-wise function. It shows one wrong way to get the speed. It ends with a moving average that you build four times.

## Three methods on every binary ufunc

A binary ufunc has a `reduce` method, an `accumulate` method and an `outer` method. Each one applies the ufunc in a different pattern.

```python
import numpy as np

x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("reduce     :", np.add.reduce(x))
print("accumulate :", np.add.accumulate(x))
print("running max:", np.maximum.accumulate(x))
print("outer      :")
print(np.multiply.outer(x[:3], x[:3]))
```

```text
reduce     : 31
accumulate : [ 3  4  8  9 14 23 25 31]
running max: [3 3 4 4 5 9 9 9]
outer      :
[[ 9  3 12]
 [ 3  1  4]
 [12  4 16]]
```

The call `np.add.reduce(x)` is `x.sum()`. It folds the array to one value. The call `np.add.accumulate(x)` is `np.cumsum(x)`. It keeps every partial result. The call `np.maximum.accumulate(x)` gives the running maximum, which has no special name in NumPy. The call `np.multiply.outer` gives every product of a pair, which is the broadcast of chapter 5 written as a method.

The methods let you build many functions from the 90 ufuncs. A running minimum, a running product and a running logical `or` are one call each.

## axis, keepdims, out and where

Every reduction takes `axis`, which can be an integer or a tuple. It takes `keepdims`, which chapter 5 used to keep the reduced axis with length 1. Every ufunc takes `out`, to write the result into an array that you made, and `where`, to skip some elements.

```python
import numpy as np

m = np.arange(6).reshape(2, 3)
print("axis 0      :", m.sum(axis=0))
print("axis 1      :", m.sum(axis=1))
print("axes (0, 1) :", m.sum(axis=(0, 1)))
print("keepdims    :", m.sum(axis=1, keepdims=True).shape)

y = np.array([1.0, -1.0, 4.0])
out = np.full(3, np.nan)
np.sqrt(y, out=out, where=y >= 0)
print("sqrt with where:", out)
```

```text
axis 0      : [3 5 7]
axis 1      : [ 3 12]
axes (0, 1) : 15
keepdims    : (2, 1)
sqrt with where: [ 1. nan  2.]
```

The last line shows the contract of `where`. The element that the condition skipped keeps its old value in `out`. If you do not pass `out`, the values of the skipped elements are undefined, so always pass it.

One more behavior has a large effect on real data. A `nan` spreads through a sum.

```python
import numpy as np

x = np.array([1.0, np.nan, 2.0])
print("sum   :", x.sum())
print("nansum:", np.nansum(x))
print("mean  :", x.mean(), "nanmean:", np.nanmean(x))
```

```text
sum   : nan
nansum: 3.0
mean  : nan nanmean: 1.5
```

The `nan` functions skip the missing values. Use them with intent. A silent `nansum` on a column that is mostly `nan` gives a number, and the number is a sum of very few values.

## Temporaries and `out=`

An expression such as `a * b + c` makes a temporary array for `a * b`, and then a second array for the sum. For small arrays this does not matter. For large arrays, each temporary is a pass over memory and an allocation. Writing into a result array that you made in advance avoids both.

```python
import timeit

import numpy as np

rng = np.random.default_rng(1)
a = rng.random(10_000_000)
b = rng.random(10_000_000)
c = rng.random(10_000_000)
out = np.empty_like(a)


def expression():
    return a * b + c


def in_place():
    np.multiply(a, b, out=out)
    np.add(out, c, out=out)
    return out


assert np.array_equal(expression(), in_place())
t_expr = min(timeit.repeat(expression, number=1, repeat=5))
t_inpl = min(timeit.repeat(in_place, number=1, repeat=5))
assert t_expr / t_inpl > 1.1
print(f"expression / in place: {t_expr / t_inpl:.2f}x")
```

```text
# output varies
expression / in place: 1.41x
```

The gain on the build machine is 1.4 times, and the assert requires 1.1. This is a modest gain, and the code is harder to read. Use `out=` when the profile shows that this line matters, and not before.

## `np.vectorize` is not vectorization

The name suggests speed. The documentation says otherwise: the function is provided for convenience, and it is a Python loop. Measure it against a plain loop and against the real vectorized form.

```python
import timeit

import numpy as np

v = np.random.default_rng(0).random(1_000_000)


def f(a):
    return a * a + 1 if a > 0.5 else a


vf = np.vectorize(f)
t_loop = min(timeit.repeat(lambda: [f(a) for a in v.tolist()], number=1, repeat=3))
t_vec = min(timeit.repeat(lambda: vf(v), number=1, repeat=3))
t_where = min(timeit.repeat(lambda: np.where(v > 0.5, v * v + 1, v), number=1, repeat=3))

assert t_vec / t_loop > 0.5          # np.vectorize is no faster than the loop
assert t_loop / t_where > 5          # the real vectorized form is
print(f"list comprehension {t_loop * 1e3:7.1f} ms")
print(f"np.vectorize       {t_vec * 1e3:7.1f} ms")
print(f"np.where           {t_where * 1e3:7.1f} ms")
```

```text
# output varies
list comprehension   166.1 ms
np.vectorize         203.4 ms
np.where             15.1 ms
```

The `np.vectorize` call is slower than the list comprehension that it replaces, on the build machine. The real vectorized form, with `np.where`, is 11 to 13 times faster than both. Use `np.vectorize` to make a Python function accept arrays in a place where speed does not matter. Do not use it to go faster.

## The function you build: a moving average

A moving average of window `w` replaces each value with the mean of the `w` values that end there. The loop version is the definition. Three others follow.

```python
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


def ma_window(x, w):
    return sliding_window_view(x, w).mean(axis=-1)


def ma_convolve(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


x = np.random.default_rng(2).random(20_000)
ref = ma_loop(x, 50)
for f in (ma_cumsum, ma_window, ma_convolve):
    print(f"{f.__name__:11s} max difference from the loop: {np.abs(f(x, 50) - ref).max():.1e}")
```

```text
ma_cumsum   max difference from the loop: 2.4e-13
ma_window   max difference from the loop: 0.0e+00
ma_convolve max difference from the loop: 3.3e-16
```

All three agree with the loop to within rounding. The version `ma_cumsum` is based on one fact: the sum of the window that ends at `i` is the cumulative sum at `i` minus the cumulative sum `w` steps earlier. This is two passes over the data, whatever the size of `w`. The version `ma_window` uses `sliding_window_view`, which makes an `(n - w + 1, w)` view without copying, as chapter 3 explained, and then takes the mean of each row. The version `ma_convolve` treats the average as a convolution with a flat kernel.

Time them on one million values with `w = 50`.

```python
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


def ma_window(x, w):
    return sliding_window_view(x, w).mean(axis=-1)


def ma_convolve(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


x = np.random.default_rng(2).random(1_000_000)
t_loop = min(timeit.repeat(lambda: ma_loop(x[:20_000], 50), number=1, repeat=3)) * 50   # scaled to 1,000,000
t = {f.__name__: min(timeit.repeat(lambda: f(x, 50), number=1, repeat=3))
     for f in (ma_cumsum, ma_window, ma_convolve)}
assert t_loop / t["ma_cumsum"] > 20
print(f"ma_loop (scaled from 20,000 values) {t_loop * 1e3:8.0f} ms")
for name, sec in t.items():
    print(f"{name:35s} {sec * 1e3:8.0f} ms")
```

```text
# output varies
ma_loop (scaled from 20,000 values)     2747 ms
ma_cumsum                                 16 ms
ma_window                                 44 ms
ma_convolve                               17 ms
```

The loop is measured on 20,000 values and multiplied by 50, so that the run is short. The vectorized versions are 60 to 170 times faster. The window version is the slowest of the three because it computes a real mean over a `w`-wide row for every output. The cumulative-sum version and the convolution are about the same speed.

## The precision trap in the fast version

The cumulative-sum version is the fastest and the shortest. It is also the one that chapter 2 warned about. Its cumulative sum grows large, and every window sum is the difference of two large numbers.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ma_cumsum(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def ma_cumsum_centered(x, w):
    m = x.mean()
    return ma_cumsum(x - m, w) + m


def ma_window(x, w):
    return sliding_window_view(x, w).mean(axis=-1)


print("float32 data, values from 10000 to 10100, one million values")
for seed in (2, 3):
    s = np.random.default_rng(seed).random(1_000_000)
    x32 = (s * 100 + 1e4).astype(np.float32)
    truth = ma_window(x32.astype(np.float64), 50)
    for f in (ma_cumsum, ma_cumsum_centered, ma_window):
        print(f"seed {seed} {f.__name__:20s} max error {np.abs(f(x32, 50) - truth).max():.3g}")
    x64 = x32.astype(np.float64)
    print(f"seed {seed} {'ma_cumsum in float64':20s} max error {np.abs(ma_cumsum(x64, 50) - truth).max():.3g}")
```

```text
float32 data, values from 10000 to 10100, one million values
seed 2 ma_cumsum            max error 207
seed 2 ma_cumsum_centered   max error 0
seed 2 ma_window            max error 0.00197
seed 2 ma_cumsum in float64 max error 0
seed 3 ma_cumsum            max error 209
seed 3 ma_cumsum_centered   max error 0.000469
seed 3 ma_window            max error 0.00186
seed 3 ma_cumsum in float64 max error 0
```

The plain cumulative-sum version has an error of about 200 on values near 10,000. That is 2 percent of the values. The window version has an error of 0.002, which is about twice the spacing of `float32` at that size.

The centered version subtracts the mean first, so the cumulative sum stays small. On seed 2 its error is exactly 0. This is a property of that data and not of the method. The centered total never leaves the range where `float32` can hold it exactly, because every value is a multiple of 2⁻¹⁰.

On seed 3 the error is 4.7e-4, which is smaller than the window version.

Read the two seeds together. Centering gives a good answer in both cases, and the exact zero on one seed is luck.

If you do not know the range of your data, use the window version or use `float64` for the cumulative sum. The last line of each seed shows the `float64` result. Its error is 0 on both seeds, for the same reason as the exact zero above. The values sit on a grid of 2⁻¹⁰, and `float64` holds their sums exactly.

## When no vectorized form exists

Some loops cannot be written with ufuncs. The usual cause is a dependency: each value needs the result for the previous value. An exponential moving average is the standard case.

```python
import time

import numpy as np


def ema_loop(x, alpha):
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x.tolist()):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


x = np.random.default_rng(2).random(1_000_000)
start = time.perf_counter()
y = ema_loop(x, 0.1)
elapsed = time.perf_counter() - start
assert np.isfinite(y[-1]) and elapsed < 10
print(f"{len(x) / elapsed / 1e6:.1f} million steps a second")
```

```text
# output varies
3.6 million steps a second
```

The loop runs at 3.6 million steps a second on the build machine. There is no `np.add.accumulate` form of this, because the weight of each step depends on the running value. Two honest options exist outside this book. A filter routine from a signal library applies this recurrence in compiled code. A compiler such as Numba turns the loop into machine code. This book did not measure either, so it makes no claim about their speed.

## What you have now

You have the three ufunc methods and four arguments: `axis`, `keepdims`, `out` and `where`. You have a moving average in four versions, with a time and an error for each. You know that `np.vectorize` is a loop. You know that `out=` saves a factor of 1.4 on large arrays. You know that the cumulative-sum trick needs centered data in `float32`.

The test that you can apply to any loop is short. If each iteration needs only its own element and its neighbors at fixed offsets, a vectorized form exists. If an iteration needs the result of the previous one, it probably does not.

Chapter 7 uses NumPy for the two jobs that most users meet next: solving linear systems and drawing random numbers that you can reproduce.
