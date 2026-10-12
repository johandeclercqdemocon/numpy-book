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

The column vector `x[:, None]` has shape `(3, 1)`, and the row vector `x[None, :]` has shape `(1, 3)`. The result `(3, 3)` holds every difference `x[i] - x[j]`.

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

The habit that follows: assert the shape of a result at a boundary. A line such as `assert r.shape == (n,)` is cheap and catches this mistake at once.

## The function you build

Given `A` with `n` points and `B` with `m` points, in `d` dimensions, build the `(n, m)` matrix of distances. Three versions follow.

```python
import numpy as np


def dist_loop(A, B):
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i, j] = np.sqrt(((a - b) ** 2).sum())
    return out


def dist_broadcast(A, B):
    """Exact. Builds an (n, m, d) temporary array."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A, B):
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

## Cost 1: time

```python
import timeit

import numpy as np


def dist_loop(A, B):
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i, j] = np.sqrt(((a - b) ** 2).sum())
    return out


def dist_broadcast(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A, B):
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
dist_loop         279.18 ms
dist_broadcast      8.76 ms
dist_dot            0.75 ms
```

The broadcast version is about 30 times faster than the loop. The dot version is about 10 times faster again. The asserts require at least 10 and at least 2. The dot version wins because a matrix product uses an optimized library routine and because it does less work: it never forms the `(n, m, d)` array.

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


def dist_broadcast(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A, B):
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


def dist_loop(A, B):
    out = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            out[i, j] = np.sqrt(((a - b) ** 2).sum())
    return out


def dist_broadcast(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot(A, B):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def dist_dot_centered(A, B):
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

The first three lines are for `float64`. The error of the plain dot version is 3.5e-4, and the mean true distance is 6.9e-3. The error is 5 percent of the answer. After centering, the error is 1.2e-10.

The `float32` lines are worse. The squared norms are about 3e8. At that size the gap between neighbor `float32` values is 32. The plain dot version reports a largest distance of 8.0. The true value is 0.0144. The answer is wrong by a factor of more than 500.

The centered version is correct. After the origin moves to the middle of the data, the numbers are small. Then `float32` has digits to spare.

Centering is the fix to remember. It costs one mean and two subtractions. It does not help when the points are spread over a large range. In that case the loss is a property of the problem and not of the method.

## Which one to use

| Version | Time | Peak memory | Precision |
|---|---|---|---|
| `dist_loop` | slowest by far | the result | exact |
| `dist_broadcast` | middle | `n * m * d` values | exact |
| `dist_dot_centered` | fastest | a few results | good when centered |

Use the dot version with centering as the default, and test it against the broadcast version on a small sample of your data. Use the broadcast version when you need the exact differences, for example when you also need the direction `a - b`. Use the loop only to check the others.

## What you have now

You have the broadcasting rule and `np.broadcast_shapes` to test it. You know that a stretch is a stride of 0 and costs nothing until a result is made. You have three distance functions, and for each one you have a measured time, a measured peak memory and a measured error.

The lesson of the chapter is that the fastest version of an array program is also the one that loses the most precision. Measure all three costs before you choose.

Chapter 6 turns to the other half of a vectorized program: the ufunc, its `reduce` and `accumulate` methods, and the cases where no vectorized form exists.
