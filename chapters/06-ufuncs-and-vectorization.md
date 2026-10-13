# Chapter 6 — Ufuncs and vectorization

> **Build** — a moving average in four versions, a loop, a cumulative-sum version, a window version and a convolution, with a measured time and a measured precision for each.
>
> **Theory kept** — a ufunc is a function that runs element by element in compiled code. It has methods (`reduce`, `accumulate`, `outer`) that apply it along an axis. Vectorization is the act of rewriting a loop as calls to ufuncs. It does not exist for every loop.
>
> **Source** — New. The ufunc methods are in the NumPy documentation. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch06_ufuncs`, which is [`examples/ch06_ufuncs.py`](../examples/ch06_ufuncs.py).

A ufunc, short for universal function, is the unit of speed in NumPy. `np.add`, `np.multiply`, `np.sqrt`, `np.maximum` and 102 others are ufuncs. The section [How it works](#how-it-works) counts them. The operators `+` and `*` call them. A ufunc takes arrays, loops over the elements in compiled code, and returns an array. It does the Python-level work once for the whole array, and not once per element.

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

#### `np.add`

```text
np.add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)
```

The ufunc adds two arrays element by element, with the broadcast rule of chapter 5. The operator `+` calls it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x1`, `x2` | required | the two inputs. They broadcast against each other | a shape that does not broadcast raises `ValueError` |
| `out` | `None` | an array to write the result into | an `out` of a type that cannot hold the result raises `UFuncTypeError`: `float64` into `int64` is refused |
| `where` | `True` | a boolean array. Elements where it is `False` are skipped | without `out`, the skipped elements have undefined values |
| `casting` | `'same_kind'` | which casts the ufunc may do to match types | `'unsafe'` allows `float64` into `int64`, and it truncates |
| `dtype` | `None` | the type of the computation and of the result | `dtype=np.int64` on two `int8` values 127 and 1 gives 128 and not -128 |
| `order` | `'K'` | the memory order of the result. `'K'` follows the inputs, `'F'` puts the first axis in consecutive memory | — |
| `subok` | `True` | with `True`, a subclass of `ndarray` in the input, such as a masked array, gives a result of that subclass. With `False`, a plain `ndarray` | with `False`, the result of a masked array has no mask |
| `signature` | `None` | the types of the inner loop, as a string such as `'dd->d'` (two `float64` in, one out). It forces one loop | `'ff->f'` on `float64` inputs computes in `float32` |

#### `np.add.reduce`

```text
np.add.reduce(array, /, axis=0, dtype=None, out=None, **kwargs)
```

The method folds the array along one axis to one value for each position of the other axes.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `array` | required | the array to fold | an empty array gives the identity of the ufunc: 0 for `np.add` |
| `axis` | `0` | the axis to fold along. With `None`, all axes | the default is 0, and `np.sum` has `None`. On the array of the next block the two give `[ 4  6 13]` and `23` |
| `dtype` | `None` | the type of the running total | — |
| `out` | `None` | an array to write the result into | — |
| `**kwargs` | — | `keepdims`, `initial` and `where`, by name. They do what they do for `np.sum` | any other name raises `TypeError` |

#### `np.add.accumulate`

```text
np.add.accumulate(array, /, axis=0, dtype=None, out=None)
```

The method returns every partial result of the fold, in order.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `array` | required | the array to scan | — |
| `axis` | `0` | the axis to scan along | the default is 0, and `np.cumsum` flattens with `None` |
| `dtype` | `None` | the type of the running total and of the result | a `float32` input keeps a `float32` total, and the drift of chapter 2 appears |
| `out` | `None` | an array to write the result into | — |

#### `np.add.outer`

```text
np.add.outer(A, B, /, **kwargs)
```

The method applies the ufunc to every pair of one element of `A` and one element of `B`. The result has the shape of `A` followed by the shape of `B`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `A`, `B` | required | the two inputs | the result has `A.ndim + B.ndim` axes, and a large pair makes a large array |
| `**kwargs` | — | the keyword arguments of the ufunc, such as `out`, `dtype` and `where` | — |

The call `np.add.reduce(x)` is `x.sum()`. It folds the array to one value. The call `np.add.accumulate(x)` is `np.cumsum(x)`. It keeps every partial result. The call `np.maximum.accumulate(x)` gives the running maximum, which has no special name in NumPy. The call `np.multiply.outer` gives every product of a pair, which is the broadcast of chapter 5 written as a method.

The methods let you build many functions from the 106 ufuncs. A running minimum, a running product and a running logical `or` are one call each.

## axis, keepdims, out and where

Every reduction takes `axis`, which can be an integer or a tuple. It takes `keepdims`, which chapter 5 used to keep the reduced axis with length 1. Every ufunc takes `out`, to write the result into an array that you made, and `where`, to skip some elements.

```python
import numpy as np

m = np.array([[3, 1, 4],
              [1, 5, 9]])
print("axis 0      :", m.sum(axis=0))
print("axis 1      :", m.sum(axis=1))
print("axes (0, 1) :", m.sum(axis=(0, 1)))
print("keepdims    :", m.sum(axis=1, keepdims=True).shape)
print("np.add.reduce(m):", np.add.reduce(m))
print("np.sum(m)       :", np.sum(m))

y = np.array([1.0, -1.0, 4.0])
out = np.full(3, np.nan)
np.sqrt(y, out=out, where=y >= 0)
print("sqrt with where:", out)
```

```text
axis 0      : [ 4  6 13]
axis 1      : [ 8 15]
axes (0, 1) : 23
keepdims    : (2, 1)
np.add.reduce(m): [ 4  6 13]
np.sum(m)       : 23
sqrt with where: [ 1. nan  2.]
```

The last line shows the contract of `where`. The element that the condition skipped keeps its old value in `out`. If you do not pass `out`, the values of the skipped elements are undefined, so always pass it.

One more behavior matters for data with missing values. A `nan` spreads through a sum.

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

#### `np.nansum`

```text
np.nansum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
```

The function adds the values of an array and treats `nan` as 0. It takes the same arguments as `np.sum`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array | an array of only `nan` gives `0.0`, which looks like a real sum |
| `axis` | `None` | the axis to add along | — |
| `dtype` | `None` | the type of the running total | — |
| `out` | `None` | an array to write the result into. Its shape must be the shape of the result | — |
| `keepdims` | `<no value>` | with `True`, the reduced axes stay with length 1 | — |
| `initial` | `<no value>` | a start value for the sum | an array of only `nan` with `initial=5` gives `5.0` |
| `where` | `<no value>` | a boolean array. Only the values where it is `True` are added | — |

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
expression / in place: 1.36x
```

The assert requires a gain of at least 1.1 times. This is a modest gain, and the code is harder to read. Use `out=` when the profile shows that this line matters, and not before.

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
list comprehension   164.9 ms
np.vectorize         198.9 ms
np.where              16.4 ms
```

#### `np.vectorize`

```text
np.vectorize(pyfunc=<no value>, otypes=None, doc=None, excluded=None, cache=False, signature=None)
```

The class wraps a Python function so that it accepts arrays. It calls the function once for each element.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `pyfunc` | `<no value>` | the Python function to wrap | it is called in a Python loop, so the speed is the speed of a loop |
| `otypes` | `None` | the output types. Without it, NumPy takes the type of the first result | `0 if v < 1 else v` on `[0.5, 2.7]` gives `[0 2]`, because the first result is an integer |
| `doc` | `None` | the docstring of the new object. With `None`, the docstring of `pyfunc` | — |
| `excluded` | `None` | a set of positions or names of arguments that are passed to `pyfunc` unchanged | without it, a list argument is split into its elements: `f(x, c)` with `c=[10, 20]` gets `10`, then `20`, and not the list |
| `cache` | `False` | with `True`, the call that finds the output type is kept, so `pyfunc` runs once on each element | without `otypes` and without `cache`, `pyfunc` runs twice on the first element. A function with side effects shows it |
| `signature` | `None` | a generalized signature such as `(n),(n)->()` for a function of arrays | a wrong signature raises `ValueError` at the call |

In the run above, the `np.vectorize` call was slower than the list comprehension that it replaces. The assert proves only that it is not more than 2 times faster. The assert also proves that the real vectorized form, with `np.where`, is more than 5 times faster than the loop. Use `np.vectorize` to make a Python function accept arrays in a place where speed does not matter. Do not use it to go faster.

## The function you build: a moving average

A moving average of window `w` replaces each value with the mean of the `w` values that end there. The loop version is the definition. Three others follow.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ma_loop(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average of window w, from the definition."""
    out = np.empty(len(x) - w + 1)
    for i in range(len(out)):
        out[i] = x[i:i + w].mean()
    return out


def ma_cumsum(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a cumulative sum."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def ma_window(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a sliding-window view."""
    return sliding_window_view(x, w).mean(axis=-1)


def ma_convolve(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average as a convolution with a flat kernel."""
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

#### `np.lib.stride_tricks.sliding_window_view`

```text
np.lib.stride_tricks.sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False)
```

The function returns a read-only view in which each row is one window of the input. It copies nothing.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the array | — |
| `window_shape` | required | the length of the window, or a tuple of lengths | a window longer than the input raises `ValueError` |
| `axis` | `None` | the axis or axes to slide along | with `None`, the window slides over all axes, and `window_shape` must have one length for each |
| `subok` | `False` | with `True`, a subclass of `ndarray`, such as a masked array, stays that subclass | with `False`, a masked array loses its mask, and the hidden values appear in the windows |
| `writeable` | `False` | with `True`, writes are allowed | a write changes every window that holds that position |

#### `np.convolve`

```text
np.convolve(a, v, mode='full')
```

The function returns the discrete convolution of two 1-D arrays. With a flat kernel, it is a moving sum.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a`, `v` | required | the signal and the kernel | when `v` is longer than `a`, NumPy swaps them, and `ma_convolve` returns 3 values for 3 inputs and `w = 5` |
| `mode` | `'full'` | `'full'` returns `n + w - 1` values. `'valid'` returns `n - w + 1` values, only where the kernel fits | `'full'` has partial windows at both ends, and they are not averages of `w` values |

The next five functions have the same two parameters. Each section below gives what goes wrong for that version.

#### `ma_loop`

```text
ma_loop(x: numpy.ndarray, w: int) -> numpy.ndarray
```

The function computes the moving average from the definition, one window at a time.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of values | the slowest version in the timing block, scaled from 20,000 values |
| `w` | required | the window length, a positive integer | a `w` larger than the input raises `ValueError` |

#### `ma_cumsum`

```text
ma_cumsum(x: numpy.ndarray, w: int) -> numpy.ndarray
```

The function computes the moving average from a cumulative sum.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of values | `float32` values from 10,000 to 10,100 give an error of 207 |
| `w` | required | the window length, a positive integer | a `w` larger than the input gives an empty array and no error |

#### `ma_cumsum_centered`

```text
ma_cumsum_centered(x: numpy.ndarray, w: int) -> numpy.ndarray
```

The function subtracts the mean first, so the cumulative sum stays small, and adds it back after.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of values | the error is 0 on seed 2 and 4.7e-4 on seed 3, for `float32` values from 10,000 to 10,100 |
| `w` | required | the window length, a positive integer | the same empty result as `ma_cumsum` for a `w` larger than the input |

#### `ma_window`

```text
ma_window(x: numpy.ndarray, w: int) -> numpy.ndarray
```

The function averages each row of a sliding-window view.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of values | the slowest of the three fast versions in the timing block |
| `w` | required | the window length, a positive integer | a `w` larger than the input raises `ValueError` |

#### `ma_convolve`

```text
ma_convolve(x: numpy.ndarray, w: int) -> numpy.ndarray
```

The function averages with a convolution and a flat kernel.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of values | — |
| `w` | required | the window length, a positive integer | a `w` larger than the input returns `len(x)` values that are not moving averages |

Time them on one million values with `w = 50`.

```python
import timeit

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ma_loop(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average of window w, from the definition."""
    out = np.empty(len(x) - w + 1)
    for i in range(len(out)):
        out[i] = x[i:i + w].mean()
    return out


def ma_cumsum(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a cumulative sum."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def ma_window(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a sliding-window view."""
    return sliding_window_view(x, w).mean(axis=-1)


def ma_convolve(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average as a convolution with a flat kernel."""
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
ma_loop (scaled from 20,000 values)     2576 ms
ma_cumsum                                 11 ms
ma_window                                 44 ms
ma_convolve                               17 ms
```

The loop is measured on 20,000 values and multiplied by 50, so that the run is short. The assert proves that the cumulative-sum version is more than 20 times faster. In the run above, the window version was the slowest of the three. It computes a real mean over a `w`-wide row for every output, which is a likely cause. This book did not test it. The cumulative-sum version and the convolution were within a factor of 2 of each other.

## The precision trap in the fast version

The cumulative-sum version is the fastest and the shortest. It is also the one that chapter 2 warned about. Its cumulative sum grows large, and every window sum is the difference of two large numbers.

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ma_cumsum(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a cumulative sum."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def ma_cumsum_centered(x: np.ndarray, w: int) -> np.ndarray:
    m = x.mean()
    return ma_cumsum(x - m, w) + m


def ma_window(x: np.ndarray, w: int) -> np.ndarray:
    """Moving average from a sliding-window view."""
    return sliding_window_view(x, w).mean(axis=-1)


print("float32 data, values from 10000 to 10100, one million values")
for seed in (2, 3):
    s = np.random.default_rng(seed).random(1_000_000)
    x32 = (s * 100 + 1e4).astype(np.float32)
    truth = ma_window(x32.astype(np.float64), 50)
    for f in (ma_cumsum, ma_cumsum_centered, ma_window):
        print(f"seed {seed} {f.__name__:20s} max error {np.abs(f(x32, 50) - truth).max():.3g}")
    centered_total = np.abs(np.cumsum(x32 - x32.mean())).max()
    print(f"seed {seed} largest centered running total {float(centered_total):.0f} (2**14 is 16384), gap at 10050: {float(np.spacing(np.float32(10050))):.6f}")
    x64 = x32.astype(np.float64)
    print(f"seed {seed} {'ma_cumsum in float64':20s} max error {np.abs(ma_cumsum(x64, 50) - truth).max():.3g}")
```

```text
float32 data, values from 10000 to 10100, one million values
seed 2 ma_cumsum            max error 207
seed 2 ma_cumsum_centered   max error 0
seed 2 ma_window            max error 0.00197
seed 2 largest centered running total 13611 (2**14 is 16384), gap at 10050: 0.000977
seed 2 ma_cumsum in float64 max error 0
seed 3 ma_cumsum            max error 209
seed 3 ma_cumsum_centered   max error 0.000469
seed 3 ma_window            max error 0.00186
seed 3 largest centered running total 30083 (2**14 is 16384), gap at 10050: 0.000977
seed 3 ma_cumsum in float64 max error 0
```

The plain cumulative-sum version has an error of about 200 on values near 10,000. That is 2 percent of the values. The window version has an error of 0.002, which is about twice the spacing of `float32` at that size.

The centered version subtracts the mean first, so the cumulative sum stays small. On seed 2 its error is exactly 0. This is a property of that data and not of the method.

Every value is a multiple of 2⁻¹⁰, which is the gap at 10,050. The `float32` format holds a multiple of 2⁻¹⁰ exactly up to 2¹⁴, which is 16,384. The block prints the largest centered total: 13,611 on seed 2, below that limit, and 30,083 on seed 3, above it.

On seed 3 the error is 4.7e-4, which is smaller than the window version.

Read the two seeds together. Centering gives a good answer in both cases, and the exact zero on one seed is luck.

If you do not know the range of your data, use the window version or use `float64` for the cumulative sum.

The last line of each seed shows the `float64` result. Its error is 0 on both seeds, for the same reason as the exact zero above. The values sit on a grid of 2⁻¹⁰, and `float64` holds their sums exactly.

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
3.5 million steps a second
```

The loop above ran at the rate that the block printed. The plain ufunc methods do not apply to this loop directly, because each step needs the result of the step before it.

A rewrite with cumulative sums and powers of `1 - alpha` exists, and this book did not test its numerical behavior. Two honest options exist outside this book. A filter routine from a signal library applies this recurrence in compiled code. A compiler such as Numba turns the loop into machine code. This book did not measure either, so it makes no claim about their speed.

## How it works

A ufunc is a table of compiled loops, one for each combination of input types. When you call it, NumPy finds the first loop in the table to which both inputs can be cast safely. It then runs that loop over the whole array. (The NumPy documentation describes this rule. This book did not measure the buffers that the cast uses.) The type letters are the names of the dtype: `b` is `int8`, `f` is `float32`, `d` is `float64`.

```python
import numpy as np

ufuncs = [name for name in dir(np) if isinstance(getattr(np, name), np.ufunc)]
print("ufuncs in the numpy namespace:", len(ufuncs))
print("compiled loops of np.add:     ", len(np.add.types))
print("first six loops:              ", np.add.types[:6])
for a, b in [(np.int8, np.int8), (np.int8, np.float32), (np.int64, np.uint64)]:
    print(f"{a.__name__:7s} + {b.__name__:8s} runs in", np.result_type(a, b))
```

```text
ufuncs in the numpy namespace: 106
compiled loops of np.add:      22
first six loops:               ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i']
int8    + int8     runs in int8
int8    + float32  runs in float32
int64   + uint64   runs in float64
```

The three results at the end are the promotion rules of chapter 2, read from the same table. The pair `int64` and `uint64` has no integer loop that holds both, so it runs as `float64`.

The two methods are also simple. `reduce` is a fold, and `accumulate` is a scan. The next block writes both with plain Python and compares them with NumPy.

```python
import functools
import itertools
import operator

import numpy as np

x = np.random.default_rng(0).random(1000)

fold = functools.reduce(operator.add, x.tolist())
scan = list(itertools.accumulate(x.tolist()))

print("reduce equals the fold:        ", np.isclose(np.add.reduce(x), fold))
print("reduce equals it exactly:      ", np.add.reduce(x) == fold)
print("difference:                    ", abs(np.add.reduce(x) - fold))
print("accumulate equals the scan:    ", np.allclose(np.add.accumulate(x), scan))
print("identity of add and multiply:  ", np.add.identity, np.multiply.identity)
print("reduce of an empty array, add: ", np.add.reduce([]))
try:
    np.maximum.reduce([])
except ValueError as e:
    print("reduce of an empty array, max: ValueError -", e)
```

```text
reduce equals the fold:         True
reduce equals it exactly:       False
difference:                     1.1368683772161603e-13
accumulate equals the scan:     True
identity of add and multiply:   0 1
reduce of an empty array, add:  0.0
reduce of an empty array, max: ValueError - zero-size array to reduction operation maximum which has no identity
```

The fold and the NumPy result agree to rounding but not exactly. `np.add.reduce` adds in pairs, as chapter 2 explained, and the Python fold adds one value at a time. The difference is of the order of 1e-13 for 1000 values.

The identity is the value of a fold over no elements: 0 for addition and 1 for multiplication. The ufunc `np.maximum` has none, so an empty reduction is an error and not a wrong number.

**The invariant.** For an associative ufunc, `reduce` gives the fold of the operator. The method `accumulate` gives every partial result. Both hold up to the order of the additions.

**The failure.** A float fold depends on the order of the additions. A Python loop and a NumPy call give different last digits. A running total in a narrow type drifts, as in chapter 2. Compare results with `np.isclose` and not with `==`.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.add` | Three methods on every binary ufunc | element-wise sum, with the broadcast rule |
| `np.add.reduce` | Three methods on every binary ufunc | fold along an axis |
| `np.add.accumulate` | Three methods on every binary ufunc | every partial result of the fold |
| `np.add.outer` | Three methods on every binary ufunc | the operator on every pair |
| `np.nansum` | axis, keepdims, out and where | sum that skips `nan` |
| `np.vectorize` | `np.vectorize` is not vectorization | a Python function applied to each element |
| `np.lib.stride_tricks.sliding_window_view` | The function you build: a moving average | read-only view of all windows |
| `np.convolve` | The function you build: a moving average | discrete convolution of two 1-D arrays |
| `ma_loop` | The function you build: a moving average | moving average from the definition |
| `ma_cumsum` | The function you build: a moving average | moving average from a cumulative sum |
| `ma_cumsum_centered` | The precision trap in the fast version | the cumulative-sum version after centering |
| `ma_window` | The function you build: a moving average | moving average from a window view |
| `ma_convolve` | The function you build: a moving average | moving average from a convolution |

## What you have now

You have the three ufunc methods and four arguments: `axis`, `keepdims`, `out` and `where`. You have a moving average in four versions, with a time and an error for each. You know that `np.vectorize` is a loop. You know that `out=` saves a factor of 1.4 on large arrays. You know that the cumulative-sum trick needs centered data in `float32`.

The test that you can apply to any loop is short. If each iteration needs only its own element and its neighbors at fixed offsets, a vectorized form exists. If an iteration needs the result of the previous one, it probably does not.

Chapter 7 uses NumPy for the two jobs that most users meet next: solving linear systems and drawing random numbers that you can reproduce.
