# Chapter 2 — Dtypes, and the answers that change silently

> **Build** — `checked_add(a, b)`, an integer add that raises an error when a result does not fit the dtype, in place of wrapping around.
>
> **Theory kept** — a dtype fixes how many bits a value has. NumPy does not check that a result fits in those bits. It keeps the bits that fit and drops the rest.
>
> **Source** — New. The promotion rules are NEP 50, which NumPy 2 implements. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch02_dtypes`, which is [`examples/ch02_dtypes.py`](../examples/ch02_dtypes.py).

Chapter 1 gave every array a dtype. This chapter shows what the dtype does to your numbers. Four things go wrong in practice.

- Integers wrap.
- Floats lose small additions.
- A sum uses a different dtype than its input.
- A mixed expression picks a dtype that you did not choose.

Each one gives a wrong answer and no error.

## What NumPy picks when you do not choose

A Python integer becomes `int64` on this machine. A Python float becomes `float64`. A mix of the two becomes `float64`.

```python
import numpy as np

print(np.array([1, 2, 3]).dtype)
print(np.array([1.0, 2]).dtype)
print(np.array([1, 2.5]).dtype)
print(np.array([1, 2, 3]).sum().dtype)
print((np.array([1, 2, 3]) / 2).dtype)
```

```text
int64
float64
float64
int64
float64
```

The last line is a common surprise. The division operator `/` always gives floats, even for integer input. Use `//` for integer division.

## Integers wrap around

An `int8` holds the values −128 to 127. NumPy does not stop you at the edge. It wraps.

```python
import numpy as np

a = np.array([100, 120, 127], dtype=np.int8)
print(a + 1)
print((a + 1).dtype)

img = np.array([200, 250, 10], dtype=np.uint8)
print(img + 100)
print(np.array([300, -5]).astype(np.uint8))
```

```text
[ 101  121 -128]
int8
[ 44  94 110]
[ 44 251]
```

The value 127 plus 1 is −128. The pixel 200 plus 100 is 44, not 300 and not 255. The call `astype(np.uint8)` turned 300 into 44 and −5 into 251. NumPy printed no message for any of these.

This is the failure that matters most for image data. A brightness boost on a `uint8` image does not clip at 255. It wraps, and the bright areas turn dark.

NumPy 2 does raise one error. A Python integer that does not fit the dtype of the array is an `OverflowError`, as the next block shows.

```python
import numpy as np

img = np.array([200, 250, 10], dtype=np.uint8)
try:
    img + 300
except OverflowError as e:
    print(type(e).__name__, "-", e)
```

```text
OverflowError - Python integer 300 out of bounds for uint8
```

The error is only for a Python integer that is too large for the dtype. It does not apply when both operands are arrays. The first example, `img + 100`, passed because 100 fits in `uint8`. The result of the add did not fit, and nobody checked.

## The function you build

The rule for a safe add is short. Do the add in a wider type. Compare each result with the limits of the narrow type. Raise an error if one result is outside the limits. `np.iinfo` gives the limits.

```python
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


pixels = np.array([200, 250, 10], dtype=np.uint8)
try:
    checked_add(pixels, np.array([100, 100, 100], dtype=np.uint8))
except OverflowError as e:
    print("refused:", e)
print(checked_add(pixels, np.array([10, 5, 100], dtype=np.uint8)))
```

```text
refused: element 0: 300 does not fit uint8
[210 255 110]
```

The function has two limits, and the code says both. It cannot check `uint64`, because no wider integer type exists in NumPy. It takes `a.dtype` as the output type, so a `uint8` plus an `int16` is checked against `uint8`. For a clip instead of an error, use `np.clip(wide, info.min, info.max)`.

## A sum has its own dtype

A sum over an `int8` array does not use `int8`. NumPy sums small integers in the platform integer, which is `int64` here. You can force the narrow type with `dtype=`, and the sum wraps.

```python
import numpy as np

a = np.array([100, 120, 127], dtype=np.int8)
print(a.sum(), a.sum().dtype)
print(a.sum(dtype=np.int8))
print(a.mean(), a.mean().dtype)
```

```text
347 int64
91
115.66666666666667 float64
```

The true sum is 347. The default sum gives 347. The forced `int8` sum gives 91, which is 347 minus 256. The mean is a `float64` and is correct.

This default is a protection and also a trap. The sum of a `uint8` image is safe. The sum of a `float32` array stays `float32`, and the next section shows that this is not safe.

## Floats lose small additions

A `float32` has 24 bits of precision. At 16,777,216, which is 2 to the power 24, the gap between two neighbor values is 2. The sum with 1 is the same number.

```python
import numpy as np

print(np.float32(16777216) + np.float32(1))
print(np.float64(16777216) + 1)
print("float32 epsilon:", np.finfo(np.float32).eps)
print("float64 epsilon:", np.finfo(np.float64).eps)
```

```text
1.6777216e+07
16777217.0
float32 epsilon: 1.1920929e-07
float64 epsilon: 2.220446049250313e-16
```

The `eps` value is the gap between 1.0 and the next float. For `float32` it is about 1.2e-7, so you get about 7 correct decimal digits. For `float64` you get about 15 or 16.

Now the practical case. Add ten million copies of 0.1 in `float32`. The exact answer is 1,000,000.015, because `float32(0.1)` is a little more than 0.1. Compare three ways to add.

```python
import numpy as np

x = np.full(10_000_000, 0.1, dtype=np.float32)
exact = float(np.float32(0.1)) * x.size

pairwise = float(x.sum())
running = float(np.cumsum(x)[-1])
wide = float(x.sum(dtype=np.float64))
for name, value in [("x.sum()", pairwise), ("cumsum[-1]", running), ("sum float64", wide)]:
    print(f"{name:12s} {value:14.3f}  relative error {abs(value - exact) / exact:.2e}")
```

```text
x.sum()         1000000.125  relative error 1.10e-07
cumsum[-1]      1087937.000  relative error 8.79e-02
sum float64     1000000.015  relative error 0.00e+00
```

The three results differ a lot. `x.sum()` has a relative error of 1e-7, which is the best a `float32` can give. The cumulative sum from `cumsum` is wrong by almost 9 percent. The cause is that a cumulative sum adds a small number to a large one. Once the total is large, each 0.1 is below the gap between neighbor values and is partly lost.

`x.sum()` is accurate because NumPy sums in pairs. It adds neighbors first, then adds the partial sums, so the numbers it adds stay close in size. `cumsum` cannot do this, because it must produce every cumulative total in order.

A pure Python loop has the same fault as `cumsum`. Chapter 6 uses this fact when it builds a moving average.

## Mixed dtypes: which one wins

When two dtypes meet, NumPy picks the smallest dtype that holds both. NumPy 2 added one rule for Python numbers. A Python `int` or `float` does not change the dtype of an array. It takes the dtype of the array, if it fits.

```python
import numpy as np

print((np.float32(3) + 3.0).dtype)
print((np.float32(3) + np.float64(3)).dtype)
print((np.int8(1) + np.int16(1)).dtype)
print((np.int64(1) + np.uint64(1)).dtype)
print((np.array([1, 2], dtype=np.uint8) + 1).dtype)
```

```text
float32
float64
int16
float64
uint8
```

Read the lines in order. A `float32` plus the Python float 3.0 stays `float32`. A `float32` plus a `float64` becomes `float64`. An `int64` plus a `uint64` has no integer type that holds both, so it becomes `float64`, and large integers lose exactness. The last line is the NumPy 2 rule: a `uint8` array plus the Python integer 1 stays `uint8`.

Older NumPy 1.x chose the dtype from the value of a Python number, so code that you find online can describe different results. Check the version of NumPy before you trust old advice.

## What you have now

You have `checked_add`, which turns a silent wrap into an error. You also have four facts that you can check on any machine. A `uint8` add wraps. A `float32` cumulative sum can be wrong by 9 percent. The call `x.sum()` does not have that fault. A mixed expression has a dtype that you can print before you trust it.

The habit that follows is short. Print `.dtype` of every array at the edge of your code, where data comes in, and again where a result goes out.

Chapter 3 moves from what an array holds to who owns it. The next surprise is not a wrong value but a value that changed in a place you did not touch.
