# Chapter 5 — Broadcasting

> **Build** — three functions that compute all distances between two sets of points, a loop version, a broadcast version and a dot-product version, with a measured memory cost and a measured precision limit for each.
>
> **Theory kept** — two shapes combine when, from the right, each pair of lengths is equal or one of them is 1. A length of 1 is stretched with a stride of 0, so the stretch costs no memory until an operation writes a result.
>
> **Source** — New. The rule is in the NumPy documentation under "Broadcasting". Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch05_broadcasting`, which is [`examples/ch05_broadcasting.py`](../examples/ch05_broadcasting.py).

Broadcasting is the rule that lets `a - b` work when `a` and `b` have different shapes. It replaces most of the loops that you write in plain Python. It also makes one new mistake easy: a result that is much larger than you meant.

This chapter states the rule, shows the stride trick behind it, and uses it to compute all pairwise distances. The distance example is the right size to show the three costs of a vectorized program: time, memory and precision.

## The rule

Write the two shapes one above the other, aligned on the right. For each column, the two lengths must be equal, or one of them must be 1. A missing length on the left counts as 1. The result has the larger length in each column.

```python
import numpy as np

print(np.broadcast_shapes((3, 1), (1, 4)))
print(np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5)))
print(np.broadcast_shapes((5, 3), (3,)))

try:
    np.ones(3) + np.ones(4)
except ValueError as e:
    print("error:", e)
```

```text
(3, 4)
(8, 7, 6, 5)
(5, 3)
error: operands could not be broadcast together with shapes (3,) (4,)
```

Read the second line with the rule. The shapes `(8, 1, 6, 1)` and `(7, 1, 5)` align as columns `8 1 6 1` over `_ 7 1 5`. The columns pair as 1 with 5, 6 with 1, 1 with 7, and 8 with nothing. The result is `(8, 7, 6, 5)`. The last line shows what happens when two lengths differ and neither is 1.

#### `np.broadcast_shapes`

```text
np.broadcast_shapes(*args)
```

The function returns the shape that the rule gives for the input shapes. It makes no array, so it is a cheap test.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `*args` | — | any number of shapes, each a tuple of integers or one integer. With one shape, the result is that shape. With none, `()` | when two lengths differ and neither is 1, it raises `ValueError` and names the shapes |

Use `None` or `np.newaxis` to add a length of 1 at any position. This is the tool that you use to line up the axes that you want.

```python
import numpy as np

x = np.arange(3)
print((x[:, None] - x[None, :]))
print((x[:, None] - x[None, :]).shape)
```

```text
[[ 0 -1 -2]
 [ 1  0 -1]
 [ 2  1  0]]
(3, 3)
```

The column vector `x[:, None]` has shape `(3, 1)`, and the row vector `x[None, :]` has shape `(1, 3)`. The result `(3, 3)` holds every difference `x[i] - x[j]`. The array `x` holds 0, 1 and 2 on purpose, so each difference is also the difference of two positions.

## A stretch costs no memory

NumPy does not copy a length-1 axis to make it longer. It sets the stride of that axis to 0, so every step reads the same element. This is the idea from chapter 1, used again.

```python
import numpy as np

row = np.arange(3.0)
stretched = np.broadcast_to(row, (1000, 3))
print("shape  ", stretched.shape)
print("strides", stretched.strides)
print("block  ", stretched.base.nbytes, "bytes")
print("result of an add:", (stretched + 1).nbytes, "bytes")
```

```text
shape   (1000, 3)
strides (0, 8)
block   24 bytes
result of an add: 24000 bytes
```

The stretched array looks like 1000 rows. It has a block of 24 bytes and a stride of 0 along the first axis. The cost appears when an operation makes a result, because the result must have all 1000 rows.

#### `np.broadcast_to`

```text
np.broadcast_to(array, shape, subok=False)
```

The function returns a read-only view with the given shape. It makes the stride of each stretched axis 0.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `array` | required | the array to stretch | — |
| `shape` | required | the target shape. The rule of this chapter must give it from the shape of `array` | a shape that the rule does not allow raises `ValueError` |
| `subok` | `False` | with `True`, a subclass of `ndarray` passes through | — |

A write to the result raises `ValueError: assignment destination is read-only`. Many of its elements share one memory location, and a write changes all of them.

## The trap: a result you did not mean

A vector of shape `(n,)` and a column of shape `(n, 1)` broadcast to `(n, n)`. NumPy gives no warning, because the rule is satisfied.

```python
import numpy as np

rng = np.random.default_rng(0)
values = rng.random(1000)
column = rng.random((1000, 1))

r = values - column
print("shape:", r.shape)
print("size: ", r.nbytes / 1e6, "MB")

X = rng.random((5, 3))
try:
    X - X.mean(axis=1)
except ValueError as e:
    print("error:", e)
print("with keepdims:", (X - X.mean(axis=1, keepdims=True)).shape)
```

```text
shape: (1000, 1000)
size:  8.0 MB
error: operands could not be broadcast together with shapes (5,3) (5,) 
with keepdims: (5, 3)
```

For `n = 1000` the mistake costs 8 MB, and nobody notices. For `n = 100,000` the same mistake needs 80 GB. The usual cause is a function that returned `(n, 1)` where you expected `(n,)`. An example is a `reshape(-1, 1)` that you added for another library.

The third part of the block is the correct way to subtract a row mean. The call `X.mean(axis=1)` has shape `(5,)`, which does not align with `(5, 3)`. The argument `keepdims=True` keeps the length-1 axis, so the shape is `(5, 1)` and the subtraction works.

#### `np.mean`

```text
np.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)
```

The function returns the mean along an axis. The method `a.mean(...)` takes the same arguments after `self`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array | an empty array gives `nan` and a warning |
| `axis` | `None` | the axis to average along. With `None`, all values | `axis=1` on a `(5, 3)` array gives shape `(5,)`, which does not align with `(5, 3)` |
| `dtype` | `None` | the type of the running total and of the result. With `None`, `float64` for an integer input, and the input type for a float input | a `float32` input gives a `float32` mean |
| `out` | `None` | an array to write the result into. Its shape must be the shape of the result | — |
| `keepdims` | `<no value>` | with `True`, the averaged axis stays with length 1 | without it, the subtraction of a row mean from the array is an error |
| `where` | `<no value>` | a boolean array. Only the values where it is `True` are averaged | if no value is `True`, the result is `nan` with a `RuntimeWarning` |

The habit that follows: assert the shape of a result at a boundary. A line such as `assert r.shape == (n,)` is cheap and catches this mistake at once.

## The function you build

Given `A` with `n` points and `B` with `m` points, in `d` dimensions, build the `(n, m)` matrix of distances. Three versions follow.

```python
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


rng = np.random.default_rng(0)
A, B = rng.random((300, 16)), rng.random((400, 16))
ref = dist_loop(A, B)
print("shape:", ref.shape)
print("broadcast max error:", np.abs(dist_broadcast(A, B) - ref).max())
print("dot       max error:", np.abs(dist_dot(A, B) - ref).max())
```

```text
shape: (300, 400)
broadcast max error: 0.0
dot       max error: 2.886579864025407e-15
```

The broadcast version is the same formula as the loop, with `A[:, None, :]` of shape `(n, 1, d)` and `B[None, :, :]` of shape `(1, m, d)`. It subtracts to `(n, m, d)`, squares, and sums over the last axis. The dot version uses the identity that the squared distance is `|a|² + |b|² − 2 a·b`. The cross term `A @ B.T` is one matrix product.

On this well-behaved input, all three agree. The dot version differs from the loop by about 3e-15, which is the rounding of `float64`.

The four versions have one signature. They differ in what goes wrong, and each of the next sections measures one of those costs.

#### `dist_loop`

```text
dist_loop(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray
```

The function returns the `(n, m)` matrix of distances with two Python loops. It is the definition.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `A` | required | `n` points as an `(n, d)` array | — |
| `B` | required | `m` points as an `(m, d)` array, with the same `d` as `A` | a different `d` raises `ValueError` |

#### `dist_broadcast`

```text
dist_broadcast(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray
```

The function returns the same matrix with one broadcast subtraction.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `A` | required | `n` points as an `(n, d)` array | the temporary array has `n * m * d` values: 2080 MB for 2000 points of 64 dimensions |
| `B` | required | `m` points as an `(m, d)` array, with the same `d` as `A` | a different `d` raises `ValueError` |

#### `dist_dot`

```text
dist_dot(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray
```

The function returns the same matrix from norms and one matrix product.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `A` | required | `n` points as an `(n, d)` array | for points in a cube of side 0.01 at 10,000: error 3.5e-4 in `float64`. In `float32` the largest distance is 8.0 where the true value is 0.0144 |
| `B` | required | `m` points as an `(m, d)` array, with the same `d` as `A` | the same loss. Centering removes it, as the section on precision shows |

## Cost 1: time

```python
import timeit

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


rng = np.random.default_rng(0)
A, B = rng.random((300, 16)), rng.random((400, 16))
t = {f.__name__: min(timeit.repeat(lambda: f(A, B), number=1, repeat=3))
     for f in (dist_loop, dist_broadcast, dist_dot)}
assert t["dist_loop"] / t["dist_broadcast"] > 10
assert t["dist_broadcast"] / t["dist_dot"] > 2
for name, sec in t.items():
    print(f"{name:15s} {sec * 1e3:9.2f} ms")
```

```text
# output varies
dist_loop          298.99 ms
dist_broadcast      13.27 ms
dist_dot             3.61 ms
```

The asserts prove two bounds. The broadcast version is at least 10 times faster than the loop. The dot version is at least 2 times faster than the broadcast version. The dot version never forms the `(n, m, d)` array, and that is a likely cause of its speed. This book did not test it.

## Cost 2: memory

The broadcast version builds an array with `n * m * d` values. For 2000 points in 64 dimensions, that is 256 million values, which is 2 GB.

```python
import tracemalloc

import numpy as np


def peak_mb(f, *args):
    tracemalloc.start()
    f(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def dist_broadcast(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact. Builds an (n, m, d) temporary array."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast and small, but loses precision when points are close together and far from the origin."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


rng = np.random.default_rng(0)
A, B = rng.random((2000, 64)), rng.random((2000, 64))
print("result alone:       ", 2000 * 2000 * 8 / 1e6, "MB")
print("broadcast peak:     ", round(peak_mb(dist_broadcast, A, B)), "MB")
print("dot peak:           ", round(peak_mb(dist_dot, A, B)), "MB")
```

```text
result alone:        32.0 MB
broadcast peak:      2080 MB
dot peak:            96 MB
```

The result itself is 32 MB. The dot version peaks at 96 MB, which is 3 times the size of the result. The broadcast version peaks at 2080 MB, which is 65 times the size of the result. The temporary array grows with `n * m * d`. At 3000 points it alone has 3000 × 3000 × 64 × 8 bytes, which is 4.6 GB.

Two remedies exist for the broadcast version. You can process `A` in blocks of rows, so that the temporary array has a fixed size. Or you can use the dot version, and read the next section first.

## Cost 3: precision

The dot version subtracts two large numbers that are almost equal. The squared norms `|a|²` and `|b|²` are large when the points are far from the origin. The squared distance is small when the points are close to each other. The difference loses digits.

```python
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


rng = np.random.default_rng(0)
P = rng.random((60, 3)) * 0.01 + 1e4        # 60 points inside a cube of side 0.01, at 10,000
ref = dist_loop(P, P)
print("mean true distance:      ", round(float(ref.mean()), 4))
print("dot            max error:", f"{np.abs(dist_dot(P, P) - ref).max():.1e}")
print("dot centered   max error:", f"{np.abs(dist_dot_centered(P, P) - ref).max():.1e}")

P32 = P.astype(np.float32)
print("float32, largest distance, true:    ", round(float(dist_broadcast(P32, P32).max()), 4))
print("float32, largest distance, dot:     ", float(dist_dot(P32, P32).max()))
print("float32, largest distance, centered:", round(float(dist_dot_centered(P32, P32).max()), 4))
```

```text
mean true distance:       0.0069
dot            max error: 3.5e-04
dot centered   max error: 1.2e-10
float32, largest distance, true:     0.0144
float32, largest distance, dot:      8.0
float32, largest distance, centered: 0.0144
```

#### `np.concatenate`

```text
np.concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting='same_kind')
```

The function joins arrays along an existing axis. The function below uses it to put the two point sets in one array before it takes the mean.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `arrays` | required | a sequence of arrays. All shapes must match, except along `axis` | other shapes raise `ValueError` and name the axis that differs |
| `axis` | `0` | the axis to join along | with `None`, the arrays are flattened first |
| `out` | `None` | an array to write the result into. Its shape must be the shape of the result | a wrong shape raises `ValueError`. You cannot give both `out` and `dtype`: that raises `TypeError` |
| `dtype` | `None` | the type of the result. With `None`, the type that holds all inputs | `float64` inputs with `dtype=np.int64` raise `TypeError` under the default `casting` |
| `casting` | `'same_kind'` | which casts are allowed when `dtype` or `out` is given. `'unsafe'` allows all | with `'unsafe'`, `1.5` becomes `1` with no warning |

#### `dist_dot_centered`

```text
dist_dot_centered(A: numpy.ndarray, B: numpy.ndarray) -> numpy.ndarray
```

The function moves the origin to the mean of both point sets, then calls `dist_dot`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `A` | required | `n` points as an `(n, d)` array | the points must be in a small range. If they are spread over a large range, centering does not help |
| `B` | required | `m` points as an `(m, d)` array, with the same `d` as `A` | a different `d` raises `ValueError` |

Centering is the fix to remember. It costs one mean and two subtractions. It does not help when the points are spread over a large range. In that case the loss is a property of the problem and not of the method.

The first three lines are for `float64`. The error of the plain dot version is 3.5e-4, and the mean true distance is 6.9e-3. The error is 5 percent of the answer. After centering, the error is 1.2e-10.

The `float32` lines are worse. The squared norms are about 3e8. At that size the gap between neighbor `float32` values is 32. The plain dot version reports a largest distance of 8.0. The true value is 0.0144. The answer is wrong by a factor of more than 500.

The centered version is correct. After the origin moves to the middle of the data, the numbers are small. Then `float32` has digits to spare.

## Which one to use

| Version | Time | Peak memory | Precision |
|---|---|---|---|
| `dist_loop` | slowest by far | the result | exact |
| `dist_broadcast` | middle | `n * m * d` values | exact |
| `dist_dot_centered` | fastest | a few results | good when centered |

Use the dot version with centering as the default, and test it against the broadcast version on a small sample of your data. Use the broadcast version when you need the exact differences, for example when you also need the direction `a - b`. Use the loop only to check the others.

## How it works

The rule is a small algorithm. Align the shapes on the right, and for each position take the one length that is not 1. If two lengths differ and neither is 1, stop with an error. The next block writes it in ten lines, and tests it against `np.broadcast_shapes` on 2000 random pairs of shapes.

```python
import numpy as np


def broadcast_shape(*shapes):
    """The broadcast rule written out: right-aligned, equal or 1."""
    n = max(len(s) for s in shapes)
    padded = [(1,) * (n - len(s)) + tuple(s) for s in shapes]
    out = []
    for lengths in zip(*padded):
        not_one = {length for length in lengths if length != 1}
        if len(not_one) > 1:
            raise ValueError(f"lengths {sorted(not_one)} cannot broadcast")
        out.append(not_one.pop() if not_one else 1)
    return tuple(out)


rng = np.random.default_rng(0)
agree = valid = 0
for _ in range(2000):
    shapes = [tuple(int(x) for x in rng.choice([1, 2, 3], size=rng.integers(0, 4))) for _ in range(2)]
    try:
        want = np.broadcast_shapes(*shapes)
    except ValueError:
        want = None
    try:
        got = broadcast_shape(*shapes)
    except ValueError:
        got = None
    agree += got == want
    valid += want is not None

print("pairs tested:       ", 2000)
print("pairs that broadcast:", valid)
print("pairs that agree:    ", agree)
```

```text
pairs tested:        2000
pairs that broadcast: 1649
pairs that agree:     2000
```

The loops over the stretched axes cost nothing in memory, because the stride is 0. The next block shows the strides that the rule gives, and then follows the shapes through one call of `dist_broadcast`.

```python
import numpy as np

col, row = np.broadcast_arrays(np.arange(3.0)[:, None], np.arange(4.0))
print("shapes: ", col.shape, row.shape)
print("strides:", col.strides, row.strides)

rng = np.random.default_rng(0)
A, B = rng.random((5, 2)), rng.random((7, 2))
steps = [
    ("A[:, None, :]", A[:, None, :]),
    ("B[None, :, :]", B[None, :, :]),
    ("difference", A[:, None, :] - B[None, :, :]),
    ("squared and summed over d", ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)),
]
for name, value in steps:
    print(f"{name:28s} {value.shape}")
```

```text
shapes:  (3, 4) (3, 4)
strides: (8, 0) (0, 8)
A[:, None, :]                (5, 1, 2)
B[None, :, :]                (1, 7, 2)
difference                   (5, 7, 2)
squared and summed over d    (5, 7)
```

Each stretched axis has stride 0, so the two inputs are read in place and the subtraction makes the one new array, `(5, 7, 2)`. This array is the cost. It has `n * m * d` values, and a sum over the last axis then shrinks it to the `(5, 7)` result.

**The invariant.** The shape of the result is the right-aligned maximum of the input shapes, and the rule makes no copy of the inputs. The block above tests the rule against NumPy on 2000 cases.

**The failure.** A shape that is almost right passes. A `(n,)` against an `(n, 1)` is allowed, and the result is `(n, n)`. The error comes only when two lengths differ and neither is 1. Test the shape of a result, because the rule cannot know what you meant.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.broadcast_shapes` | The rule | the result shape for given shapes, with no array |
| `np.broadcast_to` | A stretch costs no memory | a read-only view with the target shape |
| `np.mean` | The trap: a result you did not mean | mean along an axis, with `keepdims` |
| `dist_loop` | The function you build | distances with two Python loops |
| `dist_broadcast` | The function you build | distances with one broadcast subtraction |
| `dist_dot` | The function you build | distances from norms and a matrix product |
| `np.concatenate` | Cost 3: precision | join arrays along an existing axis |
| `dist_dot_centered` | Cost 3: precision | the dot version, with the origin in the middle |

## What you have now

You have the broadcasting rule and `np.broadcast_shapes` to test it. You know that a stretch is a stride of 0 and costs nothing until a result is made. You have three distance functions, and for each one you have a measured time, a measured peak memory and a measured error.

The lesson of the chapter is that the fastest version of an array program is also the one that loses the most precision. Measure all three costs before you choose.

Chapter 6 turns to the other half of a vectorized program: the ufunc, its `reduce` and `accumulate` methods, and the cases where no vectorized form exists.
