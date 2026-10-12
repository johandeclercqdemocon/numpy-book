# Chapter 3 — Views and copies

> **Build** — `root(a)`, a function that finds the array that owns the memory of any view, and a test that shows which operations copy and which do not.
>
> **Theory kept** — a slice is a new description of the same block. An index list or a mask makes a new block. A write through a view changes the original.
>
> **Source** — New. The rules are in the NumPy documentation under "Copies and views". Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch03_views`, which is [`examples/ch03_views.py`](../examples/ch03_views.py).

Chapter 1 showed that `a.T` and `a[:, ::2]` read the block of `a` and copy nothing. This is the reason for the speed of NumPy. It is also the reason for one class of bug: a value that changed in a place that you did not touch.

This chapter gives the rule and builds a function that applies it. It shows two bugs that come from the rule. It also measures a cost that comes from the order of memory.

## The rule

Basic slicing makes a view. Indexing with a list of integers or with a boolean mask makes a copy.

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
view = a[:, 1:3]       # basic slice
fancy = a[[0, 2]]      # list of integers
masked = a[a > 5]      # boolean mask

print("slice  shares memory:", np.shares_memory(a, view))
print("list   shares memory:", np.shares_memory(a, fancy))
print("mask   shares memory:", np.shares_memory(a, masked))
```

```text
slice  shares memory: True
list   shares memory: False
mask   shares memory: False
```

The reason is the strides from chapter 1. A slice is always a regular pattern, so one set of strides can describe it. A list of integers or a mask can pick any elements in any order, and no set of strides can describe that. NumPy must make a new block.

A write through a view changes the original.

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
view = a[:, 1:3]
view[0, 0] = -1
print(a[0])
```

```text
[ 0 -1  2  3]
```

The value at `a[0, 1]` is now −1, although the code never wrote to `a`.

## The function you build

The `.base` attribute of a view points to the array that it looks at. A view of a view points to the array one step up. Follow the chain until it ends.

```python
import numpy as np


def root(a: np.ndarray) -> np.ndarray:
    """Follow .base links to the array that owns the memory of `a`."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


a = np.arange(12).reshape(3, 4)
b = a[1:]
c = b[:, 1:]
d = c[[0]]
print("c is a view of:", root(c) is root(a))
print("d is a view of:", root(d) is root(a))
print("root of d is d:", root(d) is d)
```

```text
c is a view of: True
d is a view of: False
root of d is d: True
```

The check `isinstance(a.base, np.ndarray)` is needed. An array can be built on top of bytes, a memory-mapped file, or a buffer from another library. In those cases `.base` is not an array, and the loop must stop.

Two arrays share memory when `root(x) is root(y)`. This is a fast test. It is not exact, because two views of one block can still cover different bytes. For the exact answer, use `np.shares_memory`, and see the next block.

```python
import numpy as np

a = np.arange(10)
even, odd = a[::2], a[1::2]
print("may_share_memory:", np.may_share_memory(even, odd))
print("shares_memory:   ", np.shares_memory(even, odd))
```

```text
may_share_memory: True
shares_memory:    False
```

The function `np.may_share_memory` compares only the first and last byte that each array covers. The two interleaved slices cover the same range, so it says yes. The function `np.shares_memory` solves the exact problem and says no. The exact function can be slow for complex strides. The quick function never says no when the answer is yes.

## Bug 1: a function that changes its argument

A function that uses `-=` on its argument writes into the memory of the caller.

```python
import numpy as np


def center(x):
    x -= x.mean()
    return x


data = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
col = data[:, 0]
centered = center(col)
print("data after the call:")
print(data)
```

```text
data after the call:
[[-2. 10.]
 [ 0. 20.]
 [ 2. 30.]]
```

The caller passed one column. The column is a view, so the call changed the first column of `data`. The caller sees a different table, and the function did not return it.

The fix is one character: `x = x - x.mean()` makes a new array. The in-place operators `+=`, `-=`, `*=` and `/=` are faster and use less memory. Use them on arrays that your function made itself. Do not use them on an argument unless the name or the documentation says that the function changes it.

## Bug 2: a small view that keeps a large block alive

A view keeps its block alive. If you slice 10 rows from a 288 MB array and then delete the large array, the 288 MB stays in memory.

```python
import numpy as np

big = np.zeros((6000, 6000))
small = big[:10]
print("small is", small.nbytes, "bytes")
print("small keeps", small.base.nbytes, "bytes alive")

small = small.copy()
print("after copy, base is:", small.base)
```

```text
small is 480000 bytes
small keeps 288000000 bytes alive
after copy, base is: None
```

Use `.copy()` when you keep a small piece of a large array. The copy owns 10 rows, and the large array is free to go.

## What a copy costs, and what a bad walk costs

Chapter 1 said that a transpose is not in C order. A C-contiguous array has its last axis in consecutive memory. NumPy reads consecutive memory fast, because one read brings in a line of 64 bytes. Strided reads waste most of the line.

Measure two cases on a 6000-by-6000 `float64` array, which is 288 MB. The first is a plain copy against a copy of the transpose. The second is a sum over 600 rows against a sum over 600 columns.

```python
import timeit

import numpy as np

n = 6000
big = np.random.default_rng(0).random((n, n))


def best(f, repeat=3):
    return min(timeit.repeat(f, number=1, repeat=repeat))


t_copy = best(lambda: big.copy())
t_tcopy = best(lambda: np.ascontiguousarray(big.T))
t_row = best(lambda: [big[j, :].sum() for j in range(0, n, 10)])
t_col = best(lambda: [big[:, j].sum() for j in range(0, n, 10)])
t_ax0 = best(lambda: big.sum(axis=0))
t_ax1 = best(lambda: big.sum(axis=1))

assert t_tcopy / t_copy > 1.3
assert t_col / t_row > 3
assert 0.33 < t_ax0 / t_ax1 < 3
print(f"copy of transpose / plain copy   {t_tcopy / t_copy:5.1f}x")
print(f"600 column sums / 600 row sums   {t_col / t_row:5.1f}x")
print(f"sum(axis=0) / sum(axis=1)        {t_ax0 / t_ax1:5.1f}x")
```

```text
# output varies
copy of transpose / plain copy     2.9x
600 column sums / 600 row sums    22.3x
sum(axis=0) / sum(axis=1)          1.3x
```

The ratios above are from one run on the build machine. The asserts are the claims. The transposed copy is at least 1.3 times slower. The column loop is at least 3 times slower. The two whole-array sums are within a factor of 3 of each other.

The third line is a result that you can find surprising. The sum along axis 0 walks against the memory order, and it is still not slow. NumPy does not add one column at a time. It adds whole rows into a row of partial sums, which is the order that memory has. The built-in reductions know the layout. Your own Python loop over columns does not.

The rule that follows has two parts. Use the built-in reduction when one exists. When you must loop in Python, loop over the first axis, so that each step reads one consecutive row.

## Which operations copy

The code below tests each line of the summary table that follows. It raises an error if one line is wrong.

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
t = a.T
rows = {
    "a[1:3]":            (a[1:3], True),
    "a[:, ::2]":         (a[:, ::2], True),
    "a.T":               (t, True),
    "a[[0, 2]]":         (a[[0, 2]], False),
    "a[a > 5]":          (a[a > 5], False),
    "a.reshape(4, 3)":   (a.reshape(4, 3), True),
    "a.T.reshape(12)":   (t.reshape(12), False),
    "a.ravel()":         (a.ravel(), True),
    "a.T.ravel()":       (t.ravel(), False),
    "a.flatten()":       (a.flatten(), False),
    "a.copy()":          (a.copy(), False),
    "np.array(a)":       (np.array(a), False),
}
for name, (result, expected) in rows.items():
    shared = np.shares_memory(a, result)
    assert shared == expected, name
    print(f"{name:18s} {'view' if shared else 'copy'}")
```

```text
a[1:3]             view
a[:, ::2]          view
a.T                view
a[[0, 2]]          copy
a[a > 5]           copy
a.reshape(4, 3)    view
a.T.reshape(12)    copy
a.ravel()          view
a.T.ravel()        copy
a.flatten()        copy
a.copy()           copy
np.array(a)        copy
```

| Operation | Result |
|---|---|
| `a[1:3]`, `a[:, ::2]`, `a.T` | view |
| `a[[0, 2]]`, `a[a > 5]` | copy |
| `a.reshape(...)` | view when strides allow, copy otherwise |
| `a.ravel()` | view when the array is C-contiguous, copy otherwise |
| `a.flatten()`, `a.copy()`, `np.array(a)` | always a copy |

The call `a.reshape(...)` can change from a view to a copy when the array changes layout. Code that depends on the write-through must not rely on `reshape`. Test with `np.shares_memory` or use `np.may_share_memory`, and do not guess.

## What you have now

You have `root(a)`, and you can say which of your arrays share memory. You saw two bugs that come from views. A function changed its argument. A small view kept 288 MB alive.

You also measured the cost of a walk against the memory order. It is 3 times slower for a copy and 22 times slower for a Python loop. The built-in reductions do not have this fault.

Chapter 4 uses the other half of the rule. Indexing with lists and masks makes copies, and writing through them has its own trap.
