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

#### `np.ndarray.astype`

```text
np.ndarray.astype(self, /, dtype, order='K', casting='unsafe', subok=True, copy=True)
```

The method returns a copy of the array with another dtype. It is the tool that changed 300 into 44 above.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `dtype` | required | the new type | a narrower integer type wraps the values: 300 becomes 44 |
| `order` | `'K'` | the memory order of the result. `'K'` keeps the order of the input | — |
| `casting` | `'unsafe'` | which casts are allowed. `'safe'` allows only casts that cannot lose a value | with `'unsafe'` nothing stops 300 from becoming 44. With `'safe'` the call raises `TypeError` |
| `copy` | `True` | with `False`, the method returns the same array when the dtype already matches | with `True` the call always makes a new block |

The `casting` parameter is the guard. The call `np.array([300]).astype(np.uint8, casting='safe')` raises `TypeError: Cannot cast array data from dtype('int64') to dtype('uint8') according to the rule 'safe'`.

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

The function uses `np.iinfo` for the limits of the narrow type.

#### `np.iinfo`

```text
np.iinfo(int_type)
```

The class returns the limits of an integer type. The object has the attributes `min`, `max` and `bits`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `int_type` | required | an integer dtype, a dtype name or an integer array | a float dtype raises `ValueError`. Use `np.finfo` for floats |

#### `checked_add`

```text
checked_add(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray
```

The function adds two integer arrays and raises `OverflowError` for the first result that does not fit the dtype of `a`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | an integer array. Its dtype is the dtype of the result | a `uint64` array raises `TypeError`, because no wider type exists to check against |
| `b` | required | an integer array that broadcasts against `a` | a float array raises `TypeError` |

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

#### `np.sum`

```text
np.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
```

The function adds the values of an array along an axis. The method `a.sum(...)` takes the same arguments after `self`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array to add | — |
| `axis` | `None` | the axis to add along. `None` adds all values. A tuple adds along several axes | — |
| `dtype` | `None` | the type of the running total. With `None`, an integer input smaller than the platform integer uses `int64` | `dtype=np.int8` on the values above gives 91 and not 347, with no warning |
| `keepdims` | `<no value>` | with `True`, the reduced axes stay with length 1 | the shape of the result changes, and broadcasting differs |

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

The three results differ a lot. `x.sum()` has a relative error of 1e-7, which is the best a `float32` can give. The cumulative sum from `cumsum` is wrong by almost 9 percent, and it is too large, not too small. A cumulative sum adds a small number to a large total. The section [How it works](#how-it-works) measures what each add does.

The NumPy documentation says that `x.sum()` adds in pairs. It adds neighbors first, then adds the partial sums, so the numbers that it adds stay close in size. This book did not test that explanation. The result shows that the pairwise sum is accurate. The function `cumsum` must produce every cumulative total in order, so it cannot add in pairs.

A pure Python loop has the same fault as `cumsum`. Chapter 6 uses this fact when it builds a moving average.

Two more functions show the limits of a `float32` and the cure for the running sum.

#### `np.finfo`

```text
np.finfo(dtype)
```

The class returns the limits of a float type. The attributes `eps`, `max`, `tiny` and `bits` describe it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `dtype` | required | a float dtype, a dtype name or a float array | an integer dtype raises `ValueError`. Use `np.iinfo` for integers |

#### `np.cumsum`

```text
np.cumsum(a, axis=None, dtype=None, out=None)
```

The function returns the running total of an array. Every partial sum is in the result.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array to add | — |
| `axis` | `None` | the axis to add along. With `None`, the array is flattened first | a 2-D input becomes a 1-D result |
| `dtype` | `None` | the type of the running total and of the result | for ten million `float32` values of 0.1, the default is wrong by 8.8 percent, and `np.float64` gives 0.0 percent |

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

The documentation says that NumPy 1.x chose the dtype from the value of a Python number. This book did not install NumPy 1.x, so code that you find online can describe results that this book did not check. Check the version of NumPy before you trust old advice.

#### `np.can_cast`

```text
np.can_cast(from_, to, casting='safe')
```

The function tells you whether a cast from one dtype to another is allowed under a rule.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `from_` | required | the source dtype, or an array | — |
| `to` | required | the target dtype | — |
| `casting` | `'safe'` | the rule: `'no'`, `'equiv'`, `'safe'`, `'same_kind'` or `'unsafe'` | `'safe'` allows `int64` to `float64`, which can lose the last digits |

The last row is worth a test. The call `np.can_cast(np.int64, np.float64)` returns `True`. A `float64` has 53 bits of precision, and an `int64` has 63. The next block shows the loss.

```python
import numpy as np

print("can_cast int64 -> float64:", np.can_cast(np.int64, np.float64))
big = np.int64(2 ** 53 + 1)
print("int64 value:   ", int(big))
print("as float64:    ", int(np.float64(big)))
print("lost:          ", int(big) - int(np.float64(big)))
```

```text
can_cast int64 -> float64: True
int64 value:    9007199254740993
as float64:     9007199254740992
lost:           1
```

The word "safe" means that the type has no overflow. It does not mean that every value stays exact.

## How it works

Two facts explain every surprise in this chapter. An integer dtype does arithmetic modulo a power of 2. A float dtype stores a sign, an exponent and a fixed number of fraction bits.

**Integers.** An `int8` has 8 bits, and the add of two `int8` values keeps the low 8 bits of the result. This is arithmetic modulo 256. The bits are read as a signed number, in two's complement form. The next block shows that the bits do not change. It adds every pair of `int8` values, 65,536 pairs, and compares the result with arithmetic modulo 256.

```python
import numpy as np

v = np.arange(-128, 128, dtype=np.int8)
x, y = np.meshgrid(v, v)
with np.errstate(over="ignore"):
    got = x + y
want = ((x.astype(np.int64) + y + 128) % 256) - 128

print("pairs checked:", got.size)
print("result dtype: ", got.dtype)
print("all equal to arithmetic modulo 256:", np.array_equal(got, want))
print("bits of 127, -128, -1:", [format(int(b), "08b") for b in v[[255, 0, 127]].view(np.uint8)])
```

```text
pairs checked: 65536
result dtype:  int8
all equal to arithmetic modulo 256: True
bits of 127, -128, -1: ['01111111', '10000000', '11111111']
```

**Floats.** A `float32` has 1 sign bit, 8 exponent bits and 23 fraction bits. The value is `(-1)^sign × 2^(exponent − 127) × (1 + fraction / 2^23)`. The next block takes the bits of `0.1`, rebuilds the value with this formula, and compares it with what NumPy holds.

```python
import numpy as np

bits = int(np.array([0.1], dtype=np.float32).view(np.uint32)[0])
sign, exponent, fraction = bits >> 31, (bits >> 23) & 0xFF, bits & 0x7FFFFF
rebuilt = (-1) ** sign * 2.0 ** (exponent - 127) * (1 + fraction / 2 ** 23)

print(f"sign={sign} exponent={exponent} fraction={fraction}")
print("rebuilt value:", rebuilt)
print("NumPy value:  ", float(np.float32(0.1)))
print("equal:", rebuilt == float(np.float32(0.1)))
print("gap above 1.0:       ", np.spacing(np.float32(1.0)))
print("gap above 16777216.0:", np.spacing(np.float32(16777216.0)))
```

```text
sign=0 exponent=123 fraction=5033165
rebuilt value: 0.10000000149011612
NumPy value:   0.10000000149011612
equal: True
gap above 1.0:        1.1920929e-07
gap above 16777216.0: 2.0
```

The value is not 0.1. It is 0.10000000149011612, which is the nearest number that has 23 fraction bits. The gap between neighbor values is the value times 2⁻²³, so it doubles each time the exponent grows by 1. Above 2²⁴ the gap is 2, and this is why `16777216 + 1` rounds back.

The same rule explains the cumulative sum from earlier in this chapter. A running total in `float32` must be a multiple of the gap at its size. Each add of 0.1 rounds to the nearest multiple of that gap. The next block measures the add of `float32(0.1)` at five sizes of the total.

```python
import numpy as np

print(f"{'total':>9s} {'gap':>9s} {'0.1 became':>11s} {'change':>8s}")
for total in (1e5, 3e5, 5e5, 8e5, 1.2e6):
    t = np.float32(total)
    added = float((t + np.float32(0.1)) - t)
    print(f"{total:>9.0f} {float(np.spacing(t)):9.5f} {added:11.5f} {added / 0.1 - 1:+8.1%}")

x = np.full(10_000_000, 0.1, dtype=np.float32)
c = np.cumsum(x)
exact = float(np.float32(0.1)) * x.size
print("total passes 2**19 at step:", int(np.searchsorted(c, 2 ** 19)))
print("total passes 2**20 at step:", int(np.searchsorted(c, 2 ** 20)))
print("final relative error:      ", f"{float(c[-1]) / exact - 1:+.2%}")
```

```text
    total       gap  0.1 became   change
   100000   0.00781     0.10156    +1.6%
   300000   0.03125     0.09375    -6.2%
   500000   0.03125     0.09375    -6.2%
   800000   0.06250     0.12500   +25.0%
  1200000   0.12500     0.12500   +25.0%
total passes 2**19 at step: 5490807
total passes 2**20 at step: 9685111
final relative error:       +8.79%
```

The add of 0.1 is not lost. It is rounded, and the rounding goes in both directions. At a total of 300,000 the gap is 0.03125, and 0.1 becomes 0.09375, which is 6 percent too small. At 800,000 the gap is 0.0625, and 0.1 becomes 0.125, which is 25 percent too large.

The total passes 2¹⁹ at step 5.5 million. Both sizes above 2¹⁹ in the table give an add that is 25 percent too large. Together with the steps below that size, this gives a final result that is 8.8 percent too large. The block did not measure each of the ten million adds, so the split between the ranges is not measured.

**The invariant.** For an integer dtype, every result is the true result modulo 2 to the number of bits. For a float dtype, every result is the nearest value that has the fraction bits of the type. You can test both statements, and the two blocks above do so.

**The failure.** Each wrong answer in this chapter has this cause. A `uint8` pixel of 200 plus 100 is 300 modulo 256, which is 44. A `float32` running total rounds every add to a multiple of its gap. The cumulative sum of ten million 0.1 values is 8.8 percent too large. NumPy does not report either one, because both results are correct under the rules of the dtype.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.ndarray.astype` | Integers wrap around | a copy with another dtype, under a casting rule |
| `np.iinfo` | The function you build | the limits of an integer type |
| `checked_add` | The function you build | integer add that raises `OverflowError` and never wraps |
| `np.sum` | A sum has its own dtype | total along an axis, in a type that you can set |
| `np.finfo` | Floats lose small additions | the limits and precision of a float type |
| `np.cumsum` | Floats lose small additions | running total, in a type that you can set |
| `np.can_cast` | Mixed dtypes: which one wins | whether a cast is allowed under a rule |

## What you have now

You have `checked_add`, which turns a silent wrap into an error. You also have four facts that you can check on any machine. A `uint8` add wraps. A `float32` cumulative sum can be wrong by 9 percent. The call `x.sum()` does not have that fault. A mixed expression has a dtype that you can print before you trust it.

The habit that follows is short. Print `.dtype` of every array at the edge of your code, where data comes in, and again where a result goes out.

Chapter 3 moves from what an array holds to who owns it. The next surprise is not a wrong value but a value that changed in a place you did not touch.
