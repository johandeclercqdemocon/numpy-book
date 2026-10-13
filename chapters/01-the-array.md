# Chapter 1 — The array

> **Build** — a function `describe(a)` that prints the memory layout of any array, and a measurement of how much faster an array sum is than a list sum.
>
> **Theory kept** — an array is one block of memory plus three numbers per axis: the shape, the dtype and the strides. Nothing else.
>
> **Source** — New. The behavior is from the NumPy documentation for `numpy.ndarray`. Every number is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch01_array`, which is [`examples/ch01_array.py`](../examples/ch01_array.py).

A NumPy array is a block of memory and a description of how to read it. The block holds raw values, side by side, with no gaps. The description tells NumPy where one row ends and the next row starts. When you know these two facts, most of NumPy is no surprise.

This chapter shows the description, measures what it buys, and shows one case where it buys nothing. Each function that the chapter uses has its printed signature and a parameter table at the place where it first appears. The section [How it works](#how-it-works) shows where each number in the description comes from.

## The description: shape, dtype, strides

An array has three attributes that fix its layout.

- `shape` is a tuple with the length of each axis.
- `dtype` is the type of each value. It fixes the size of one value in bytes, which is `itemsize`.
- `strides` is a tuple with the number of bytes to jump to move one step along each axis.

Make a 3-by-4 array of 32-bit integers and print the three attributes.

```python
import numpy as np

a = np.arange(12, dtype=np.int32).reshape(3, 4)
print(a)
print("shape   ", a.shape)
print("itemsize", a.itemsize)
print("strides ", a.strides)
print("nbytes  ", a.nbytes)
```

```text
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
shape    (3, 4)
itemsize 4
strides  (16, 4)
nbytes   48
```

Read the strides as a rule. To move one column to the right, jump 4 bytes, which is one `int32`. To move one row down, jump 16 bytes, which is 4 values of 4 bytes. The array has 12 values of 4 bytes each, so `nbytes` is 48.

The address of the value at row `i` and column `j` is therefore the start of the block plus `i * 16 + j * 4`. There is no other rule. Slicing, transposing and reshaping all work the same way. They change these numbers and do not touch the block.

The function `np.arange` made the values of this array. Its signature and parameters follow.

#### `np.arange`

```text
np.arange(start_or_stop, /, stop=None, step=1, *, dtype=None, device=None, like=None)
```

The function returns an array of evenly spaced values. It has the same role as the Python `range`, and it accepts floats.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `start_or_stop` | required | with one argument, the stop value and the start is 0. With two, the start | — |
| `stop` | `None` | the end of the range. The documented intent is that it is not included | for float steps the last value can be above it (next section) |
| `step` | `1` | the distance between values. A negative step counts down | a float step gives a count that depends on rounding |
| `dtype` | `None` | the type of each value. With `None`, NumPy takes an integer type if all arguments are integers, and `float64` if not | a range that is too large for the dtype wraps without an error: 300 values in `int8` end at 43 |
| `device`, `like` | — | for use with other array libraries | — |

The last row is a short form: both parameters have the default `None`, and you leave them unset.

## Making arrays

Three more functions make most of the arrays that you need. Each gives the shape and the dtype that you ask for, and nothing else.

#### `np.zeros`

```text
np.zeros(shape, dtype=None, order='C', *, device=None, like=None)
```

The function returns a new array of the given shape, filled with zeros. Its relatives `np.ones`, `np.full` and `np.empty` take the same first arguments. The function `np.empty` does not set the values, so the block holds whatever was in that memory before. Never read an array from `np.empty` before you write to it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `shape` | required | an integer or a tuple of integers. The value `3` gives 3 values. The value `(2, 3)` gives 2 rows of 3 | a bare `2, 3` is an error: the second number is taken as the dtype |
| `dtype` | `None` | the type of each value. With `None`, the type is `float64` | the default is a float, not an integer |
| `order` | `'C'` | `'C'` puts the last axis in consecutive memory, `'F'` puts the first axis there | — |

#### `np.linspace`

```text
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)
```

The function returns `num` values that are evenly spaced between `start` and `stop`. Use it when you know how many values you want. Use `np.arange` when you know the step.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `start`, `stop` | required | the first value and the last value of the range | — |
| `num` | `50` | the number of values | — |
| `endpoint` | `True` | with `False`, the range does not include `stop`, and the step is `(stop - start) / num` | the step changes when you switch it |
| `retstep` | `False` | with `True`, the function returns the pair `(array, step)` | the result is a tuple, not an array |
| `dtype` | `None` | the type of the result. With `None`, a float | an integer dtype truncates the values |

The two functions differ in one important way. The step of `np.arange` is given, and the count is computed from floats. The count of `np.linspace` is given, and the step is computed.

```python
import numpy as np

r = np.arange(1, 1.3, 0.1)
print("arange(1, 1.3, 0.1) has", len(r), "values")
print("last value:", r[-1], "and the stop was 1.3")
print("last value is above the stop:", r[-1] > 1.3)
print("quotient (stop - start) / step:", (1.3 - 1) / 0.1)

s = np.linspace(1, 1.3, 4)
print("linspace(1, 1.3, 4):", s)
```

```text
arange(1, 1.3, 0.1) has 4 values
last value: 1.3000000000000003 and the stop was 1.3
last value is above the stop: True
quotient (stop - start) / step: 3.0000000000000004
linspace(1, 1.3, 4): [1.  1.1 1.2 1.3]
```

The range `1` to `1.3` with step `0.1` has the quotient `(1.3 - 1) / 0.1`. In floats this is 3.0000000000000004, and NumPy rounds up. It makes 4 values, and the last one is larger than the stop. When you need a known number of values, use `np.linspace`.

#### `np.array`

```text
np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, ndmax=0, like=None)
```

The function builds an array from Python objects. It is the usual way to enter data by hand.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `object` | required | a list, a tuple, a number or another array. Equal-length nested lists make a 2-D array | unequal rows are an error in NumPy 2 |
| `dtype` | `None` | the type of each value. With `None`, NumPy takes the smallest type that holds all the values | a mix of numbers and text becomes text |
| `copy` | `True` | `True` always copies. `None` copies only when needed. `False` raises an error if a copy cannot be avoided | `True` costs memory for a large input |
| `order` | `'K'` | the memory order of the result. `'K'` keeps the order of the input | — |
| `ndmin` | `0` | the least number of axes. NumPy puts length-1 axes on the left to reach it | — |

The function picks the dtype for you, and sometimes the choice is not the one that you expect.

```python
import numpy as np

print(np.array([1, 2, 3]).dtype)
print(np.array([True, 2]).dtype)
print(np.array([1, 2.5]).dtype)
print(np.array([1, "a"]).dtype)

try:
    np.array([[1, 2], [3]])
except ValueError as e:
    print("ValueError:", str(e).split(".")[0])
```

```text
int64
int64
float64
<U21
ValueError: setting an array element with a sequence
```

A mix of an integer and a string becomes an array of strings, `<U21`, and the number 1 becomes the text `'1'`. A list of unequal rows is an error in NumPy 2. NumPy 1.x made an array of objects for the same call. Check the dtype of any array that you build from mixed data.

## A transpose copies nothing

Transpose the array and print its layout again.

```python
import numpy as np

a = np.arange(12, dtype=np.int32).reshape(3, 4)
b = a.T
print("a strides", a.strides, "shape", a.shape)
print("b strides", b.strides, "shape", b.shape)
print("same memory:", np.shares_memory(a, b))
print("b is C-contiguous:", b.flags.c_contiguous)
```

```text
a strides (16, 4) shape (3, 4)
b strides (4, 16) shape (4, 3)
same memory: True
b is C-contiguous: False
```

The shape went from `(3, 4)` to `(4, 3)`, and the strides swapped from `(16, 4)` to `(4, 16)`. The 48 bytes did not move. The function `np.shares_memory` confirms that `a` and `b` read the same block.

#### `np.shares_memory`

```text
np.shares_memory(a, b, /, max_work=-1)
```

The function returns `True` if the two arrays have at least one byte of memory in common.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a`, `b` | required | the two arrays to compare | — |
| `max_work` | `-1` | how much effort to spend. `-1` solves the problem exactly. `0` compares only the first and last byte of each array | the exact answer can take very long for complex strides |

The last line of the block above shows the price of a transpose. A C-contiguous array has its last axis in consecutive memory, so a walk along the last axis reads memory in order. The transpose does not have this property. A loop over its last axis jumps 16 bytes at each step. Chapter 3 measures what that costs.

## The function you build

Put the three attributes and the two flags that matter into one function. You will use it in every later chapter.

```python
import numpy as np


def describe(a: np.ndarray) -> str:
    """One line that tells the whole layout of an array."""
    return (f"shape={a.shape} dtype={a.dtype} itemsize={a.itemsize} "
            f"strides={a.strides} nbytes={a.nbytes} "
            f"c_contiguous={a.flags.c_contiguous} owns={a.flags.owndata}")


a = np.arange(12, dtype=np.int32).reshape(3, 4)
print(describe(a))
print(describe(a.T))
print(describe(a[:, ::2]))
```

```text
shape=(3, 4) dtype=int32 itemsize=4 strides=(16, 4) nbytes=48 c_contiguous=True owns=False
shape=(4, 3) dtype=int32 itemsize=4 strides=(4, 16) nbytes=48 c_contiguous=False owns=False
shape=(3, 2) dtype=int32 itemsize=4 strides=(16, 8) nbytes=24 c_contiguous=False owns=False
```

#### `describe`

```text
describe(a: numpy.ndarray) -> str
```

The function reads attributes only, and it never copies the array.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | any `ndarray`, of any shape and dtype | a Python list has no `.shape`, so the call raises `AttributeError` |

The `owns` field is `True` when the array owns its block. It is `False` when the array only looks at the block of another array. The slice `a[:, ::2]` takes every second column. It has strides `(16, 8)`, so it reads the same block with a step of 8 bytes.

The first line is a surprise. The array `a` itself has `owns=False`. The call `np.arange(12)` made the block, and `reshape` returned a new view of that block. The variable `a` is a view, and the block belongs to an array that no name points to. Chapter 3 shows why this matters.

## Why a sum over an array is fast

A Python list holds pointers. Each pointer leads to a separate integer object somewhere in memory. A sum over the list must follow each pointer, check the type of the object, and add. An array holds the values themselves, so a sum is one tight loop over consecutive bytes.

(This describes the CPython object model. The blocks below measure the time and the memory that follow from it. They do not inspect the interpreter. Appendix A lists this statement.)

Measure both. The code below sums one million integers in each form and keeps the best of five runs.

```python
import timeit

import numpy as np

n = 1_000_000
as_list = list(range(n))
as_array = np.arange(n)

t_list = min(timeit.repeat(lambda: sum(as_list), number=5, repeat=5)) / 5
t_array = min(timeit.repeat(lambda: as_array.sum(), number=5, repeat=5)) / 5
ratio = t_list / t_array
assert sum(as_list) == int(as_array.sum())
assert ratio > 5, "chapter 1 claims an array sum is at least 5 times faster"
print(f"list sum   {t_list * 1e3:8.3f} ms")
print(f"array sum  {t_array * 1e3:8.3f} ms")
print(f"ratio      {ratio:8.1f}x")
```

```text
# output varies
list sum      9.604 ms
array sum     0.287 ms
ratio          33.4x
```

The numbers above are from one run on a shared machine. Yours will differ. The `assert` is the claim: the array sum is at least 5 times faster. The assert fails if the ratio falls below 5 on your machine.

Memory shows the same effect. The list holds 8-byte pointers and, in addition, one 28-byte object for each integer above 256.

```python
import sys

import numpy as np

n = 1_000_000
as_list = list(range(n))
list_bytes = sys.getsizeof(as_list) + sum(sys.getsizeof(x) for x in as_list)
array_bytes = np.arange(n).nbytes
print("list  ", list_bytes, "bytes")
print("array ", array_bytes, "bytes")
print("ratio ", round(list_bytes / array_bytes, 1))
```

```text
list   36000056 bytes
array  8000000 bytes
ratio  4.5
```

The list needs 4.5 times the memory of the array for the same one million integers. The ratio depends on the dtype. An `int8` array of the same length is 36 times smaller than the list.

## A case where the array is slower

The array has a fixed cost for each call. NumPy must check the type, allocate a result and start the loop. For a very small input, this cost is larger than the work.

```python
import timeit

import numpy as np

small_list = [1, 2, 3]
small_array = np.array(small_list)
t_list = min(timeit.repeat(lambda: sum(small_list), number=20000, repeat=5)) / 20000
t_array = min(timeit.repeat(lambda: small_array.sum(), number=20000, repeat=5)) / 20000
print("array is slower for 3 values:", t_array > t_list)
assert t_array > t_list
```

```text
array is slower for 3 values: True
```

The list wins for three values. Use NumPy when you have thousands of values or more, or when the shape of the problem is a table. For a handful of numbers, use a list and do not apologize.

## How it works

Everything above follows from one fact: the array is a block of bytes and three attributes that say how to read it. This section opens the block and checks each claim against the bytes.

An array of four 16-bit integers is eight bytes in a row.

```python
import numpy as np

a = np.arange(4, dtype=np.int16)
print(a.tobytes())
print(list(a.tobytes()))
```

```text
b'\x00\x00\x01\x00\x02\x00\x03\x00'
[0, 0, 1, 0, 2, 0, 3, 0]
```

Each integer uses two bytes, and the low byte comes first. This is the little-endian order of most computers. The block has no header, no length and no type. The array object holds those facts separately, and `strides` is how it finds a value in the block.

The rule for the address is general. For the index `(i0, i1, ..., ik)`, the byte offset is the sum of `i_n * strides[n]`. The next block tests this rule on a 3-axis array. It reads every value from the raw bytes with the formula, and compares it with the value that NumPy returns.

```python
import itertools

import numpy as np

a = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
raw = a.tobytes()


def read_at(index):
    offset = sum(i * s for i, s in zip(index, a.strides))
    return int.from_bytes(raw[offset:offset + a.itemsize], "little", signed=True)


every = list(itertools.product(*[range(n) for n in a.shape]))
print("strides:", a.strides)
print("values checked:", len(every))
print("all equal:", all(read_at(ix) == a[ix] for ix in every))
```

```text
strides: (24, 8, 2)
values checked: 24
all equal: True
```

The strides come from the shape. The last axis has stride 2, which is the item size. Each earlier stride is the next stride times the next length: 2 × 4 = 8, and 8 × 3 = 24. This is C order, also called row-major order. In C order, the last axis changes fastest in memory.

The other common order is Fortran order, or column-major order. The first axis changes fastest. NumPy can hold the same values in either order, and only the strides differ.

```python
import numpy as np

c = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
f = np.asfortranarray(c)

print("C strides:", c.strides, "C-contiguous:", c.flags.c_contiguous)
print("F strides:", f.strides, "F-contiguous:", f.flags.f_contiguous)
print("same values:", np.array_equal(c, f))
print("same memory:", np.shares_memory(c, f))
```

```text
C strides: (24, 8, 2) C-contiguous: True
F strides: (2, 4, 12) F-contiguous: True
same values: True
same memory: False
```

The Fortran strides `(2, 4, 12)` run the other way: 2, then 2 × 2 = 4, then 4 × 3 = 12. The two arrays hold equal values in two blocks, so `asfortranarray` made a copy.

**The invariant.** The byte offset of an element is always the sum of index times stride, for any shape, any order and any view. A test of this is the block above.

**The failure.** Code can assume a C-ordered block and receive another order. It then reads the right values slowly. Or it passes a wrong pointer to a library that expects one order. Chapter 3 measures the slow case. The default order is C.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.arange` | The description: shape, dtype, strides | evenly spaced values, with a given step |
| `np.zeros` | Making arrays | an array of zeros of a given shape and dtype |
| `np.linspace` | Making arrays | a given count of evenly spaced values |
| `np.array` | Making arrays | an array from Python objects |
| `np.shares_memory` | A transpose copies nothing | exact test for a shared byte |
| `describe` | The function you build | one line with the layout of an array |

## What you have now

You can read any array as a block plus `(shape, dtype, strides)`, and you know where each stride comes from. You can make arrays with four functions, and you know what their parameters do and where each one goes wrong. You have a `describe` function that prints the layout.

You also measured three results. The array sum is at least 5 times faster, which the assert proves. The memory is 4.5 times smaller. Three values are faster in a list.

Chapter 2 looks at the `dtype` field, which decides what a value is, and at the ways it silently changes the answer.
