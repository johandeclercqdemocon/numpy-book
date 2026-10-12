# Chapter 1 — The array

> **Build** — a function `describe(a)` that prints the memory layout of any array, and a measurement of how much faster an array sum is than a list sum.
>
> **Theory kept** — an array is one block of memory plus three numbers per axis: the shape, the dtype and the strides. Nothing else.
>
> **Source** — New. The behavior is from the NumPy documentation for `numpy.ndarray`. Every number is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch01_array`, which is [`examples/ch01_array.py`](../examples/ch01_array.py).

A NumPy array is a block of memory and a description of how to read it. The block holds raw values, side by side, with no gaps. The description tells NumPy where one row ends and the next row starts. When you know these two facts, most of NumPy is no surprise.

This chapter shows the description, measures what it buys, and shows one case where it buys nothing.

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

The address of the value at row `i` and column `j` is therefore the start of the block plus `i * 16 + j * 4`. There is no other rule. A slice, a transpose and a reshape all work the same way. They change these numbers and do not touch the block.

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

The shape went from `(3, 4)` to `(4, 3)`, and the strides swapped from `(16, 4)` to `(4, 16)`. The 48 bytes did not move. `np.shares_memory` confirms that `a` and `b` read the same block.

The last line shows the price. A C-contiguous array has its last axis in consecutive memory, so a walk along the last axis reads memory in order. The transpose does not have this property. A loop over its last axis jumps 16 bytes at each step. Chapter 3 measures what that costs.

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

The `owns` field is `True` when the array owns its block. It is `False` when the array only looks at the block of another array. The slice `a[:, ::2]` takes every second column. It has strides `(16, 8)`, so it reads the same block with a step of 8 bytes.

The first line is a surprise. The array `a` itself has `owns=False`. The call `np.arange(12)` made the block, and `reshape` returned a new view of that block. The variable `a` is a view, and the block belongs to an array that no name points to. Chapter 3 shows why this matters.

## Why a sum over an array is fast

A Python list holds pointers. Each pointer leads to a separate integer object somewhere in memory. A sum over the list must follow each pointer, check the type of the object, and add. An array holds the values themselves, so a sum is one tight loop over consecutive bytes.

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
list sum      9.2 ms
array sum     0.7 ms
ratio        13x
```

The numbers above are from one run on the machine that built this book. Yours will differ. The `assert` is the claim: the array sum is at least 5 times faster. The run on the build machine gave a ratio of about 13. The assert fails if the ratio falls below 5 on yours.

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

## What you have now

You can read any array as a block plus `(shape, dtype, strides)`. You have a `describe` function that prints this. You measured three results. The array sum is about 13 times faster on one machine. The memory is 4.5 times smaller. Three values are faster in a list.

Chapter 2 looks at the `dtype` field, which decides what a value is, and at the ways it silently changes the answer.
