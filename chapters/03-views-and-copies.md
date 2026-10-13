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

a = np.array([[3, 1, 4, 1],
              [5, 9, 2, 6],
              [5, 3, 5, 8]])
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

a = np.array([[3, 1, 4, 1],
              [5, 9, 2, 6],
              [5, 3, 5, 8]])
view = a[:, 1:3]
view[0, 0] = -1
print(a[0])
```

```text
[ 3 -1  4  1]
```

The value at `a[0, 1]` is now −1, although the code never wrote to `a`.

## The function you build

The `.base` attribute of a view points to the array that owns the block. NumPy sets it to the owner directly, even for a view of a view, so one step is enough. The loop below follows the chain to its end anyway. A block can belong to an object that is not an array, such as `bytes`.

```python
import numpy as np


def root(a: np.ndarray) -> np.ndarray:
    """Follow .base links to the array that owns the memory of `a`."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


a = np.array([[3, 1, 4, 1],
              [5, 9, 2, 6],
              [5, 3, 5, 8]])
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

#### `root`

```text
root(a: numpy.ndarray) -> numpy.ndarray
```

The function returns the array that owns the memory of `a`. If `a` owns its memory, it returns `a`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | any `ndarray`, a view or an owner | an array built on `bytes` has a base that is not an array, so `root` returns that array and not the `bytes` object |

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

The function `np.may_share_memory` compares only the first and last byte that each array covers. The two interleaved slices cover the same range, so it says yes. The function `np.shares_memory` solves the exact problem and says no. The exact function can be slow for complex strides.

#### `np.may_share_memory`

```text
np.may_share_memory(a, b, /, max_work=0)
```

The function is the quick version of `np.shares_memory`. It tests only whether the memory ranges of the two arrays overlap.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a`, `b` | required | the two arrays to compare | — |
| `max_work` | `0` | the effort. `0` compares only the first and last byte of each array | the interleaved slices `a[::2]` and `a[1::2]` give `True`, and `np.shares_memory` gives `False` |

The quick function can say yes when the exact answer is no. The next block tests the other direction on 5000 random pairs of slices: does the quick function ever say no when the exact answer is yes?

```python
import numpy as np

rng = np.random.default_rng(0)
base = np.arange(60)


def random_slice():
    start = int(rng.integers(0, 60))
    stop = int(rng.integers(start, 61))
    step = int(rng.integers(1, 6))
    return base[start:stop:step]


both = quick_yes_exact_no = quick_no_exact_yes = 0
for _ in range(5000):
    a, b = random_slice(), random_slice()
    if a.size == 0 or b.size == 0:
        continue
    exact, quick = np.shares_memory(a, b), np.may_share_memory(a, b)
    both += exact and quick
    quick_yes_exact_no += quick and not exact
    quick_no_exact_yes += exact and not quick

print("both say yes:                    ", both)
print("quick says yes, exact says no:   ", quick_yes_exact_no)
print("quick says no, exact says yes:   ", quick_no_exact_yes)
```

```text
both say yes:                     1634
quick says yes, exact says no:    581
quick says no, exact says yes:    0
```

The third count is 0, as the documentation promises. A `False` from the quick function is certain. A `True` is not.

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

Chapter 1 said that a transpose is not in C order. A C-contiguous array has its last axis in consecutive memory. NumPy reads consecutive memory fast. A likely cause is the line of 64 bytes that one read brings in. A strided read wastes most of that line. This book did not test the explanation.

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
copy of transpose / plain copy     2.7x
600 column sums / 600 row sums    21.8x
sum(axis=0) / sum(axis=1)          1.4x
```

The ratios above are from one run on a shared machine. The asserts are the claims. The transposed copy is at least 1.3 times slower. The column loop is at least 3 times slower. The two whole-array sums are within a factor of 3 of each other.

The third line is a result that you can find surprising. The sum along axis 0 walks against the memory order, and it is still not slow. NumPy does not add one column at a time. It adds whole rows into a row of partial sums, which is the order that memory has. The built-in reductions know the layout. Your own Python loop over columns does not.

#### `np.ascontiguousarray`

```text
np.ascontiguousarray(a, dtype=None, *, like=None)
```

The function returns an array in C order. It copies only when the input is not already C-contiguous.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | any array or array-like | for an input that is already contiguous, the result is the input itself, and a write reaches the original |
| `dtype` | `None` | the type of the result. With `None`, the dtype of the input | — |
| `like` | `None` | an array of another array library. The result is then an array of that library | a list raises `TypeError` |

The rule that follows has two parts. Use the built-in reduction when one exists. When you must loop in Python, loop over the first axis, so that each step reads one consecutive row.

## Which operations copy

Two methods decide the first rows of the table, so their signatures come first.

#### `np.ndarray.reshape`

```text
np.ndarray.reshape(self, /, *shape, order='C', copy=None)
```

The method returns an array with the same values and a new shape. It returns a view when the strides allow it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `*shape` | — | the new shape, as a tuple or as separate integers. One length can be `-1`, and NumPy computes it | the product of the new lengths must equal the size, or the call raises `ValueError` |
| `order` | `'C'` | the order in which values are read and placed. `'C'` means that the last axis changes fastest | `'F'` on a C array reorders the values, and it can force a copy |
| `copy` | `None` | `None` gives a view when possible and a copy when not. `True` always copies. `False` raises `ValueError` if a copy cannot be avoided | with `None` you do not know if a later write reaches the original. With `False` the transposed case raises `Unable to avoid creating a copy while reshaping.` |

#### `np.ndarray.copy`

```text
np.ndarray.copy(self, /, order='C')
```

The method returns a new array with its own block.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `order` | `'C'` | the memory order of the copy. `'F'` is Fortran order. `'A'` is the order of the input. `'K'` is as close to the input as possible | the function `np.copy` has the default `'K'`, so `a.T.copy()` has strides `(24, 8)` and `np.copy(a.T)` has strides `(8, 32)` for the array in the next block |

The code below tests each line of the summary table that follows. It raises an error if one line is wrong.

```python
import numpy as np

a = np.array([[3, 1, 4, 1],
              [5, 9, 2, 6],
              [5, 3, 5, 8]])
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
print("a.T.copy() strides:", a.T.copy().strides)
print("np.copy(a.T) strides:", np.copy(a.T).strides)
for name, (result, expected) in rows.items():
    shared = np.shares_memory(a, result)
    assert shared == expected, name
    print(f"{name:18s} {'view' if shared else 'copy'}")
```

```text
a.T.copy() strides: (24, 8)
np.copy(a.T) strides: (8, 32)
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

## How it works

A slice makes a new array object, not a new block. The object holds a pointer into the block, a shape, strides and a few flags. Making it costs the same for a small array and for a large one.

The next block measures this. It makes a view of 5000 rows of a 288 MB array and a copy of the whole array.

```python
import timeit

import numpy as np

big = np.zeros((6000, 6000))
t_view = min(timeit.repeat(lambda: big[:5000], number=1000, repeat=3)) / 1000
t_copy = min(timeit.repeat(lambda: big.copy(), number=1, repeat=3))
assert t_copy / t_view > 1000
print("view is a new object over the same block:", big[:5000].base is big)
print("view costs under a millisecond:", t_view < 1e-3)
print("a copy costs at least 1000 times a view:", t_copy / t_view > 1000)
```

```text
view is a new object over the same block: True
view costs under a millisecond: True
a copy costs at least 1000 times a view: True
```

The flags show what the object knows about its block. The `OWNDATA` flag tells who frees the memory. The `WRITEABLE` flag tells if a write is allowed.

```python
import numpy as np

a = np.array([[3, 1, 4, 1],
              [5, 9, 2, 6],
              [5, 3, 5, 8]])
cases = {
    "a": a,
    "a.T": a.T,
    "a[::2]": a[::2],
    "broadcast_to": np.broadcast_to(np.arange(3.0), (2, 3)),
}
print(f"{'':14s} {'C':>5s} {'F':>5s} {'write':>6s} {'owns':>6s}")
for name, v in cases.items():
    f = v.flags
    print(f"{name:14s} {f.c_contiguous!s:>5s} {f.f_contiguous!s:>5s} {f.writeable!s:>6s} {f.owndata!s:>6s}")

try:
    cases["broadcast_to"][0, 0] = 1.0
except ValueError as e:
    print("write to broadcast_to:", e)

frozen = a.copy()
frozen.setflags(write=False)
try:
    frozen[0, 0] = 1
except ValueError as e:
    print("write to frozen copy: ", e)
```

```text
                   C     F  write   owns
a               True False   True   True
a.T            False  True   True  False
a[::2]         False False   True  False
broadcast_to   False False  False  False
write to broadcast_to: assignment destination is read-only
write to frozen copy:  assignment destination is read-only
```

The result of `np.broadcast_to` is read-only. The NumPy documentation gives the reason: many of its elements share one memory location. A write to one element changes all of them.

You can protect your own arrays in the same way with `setflags(write=False)`. A function that must not change its argument can take a read-only view. A write then fails at once with an error. It does not give a wrong number later.

**The invariant.** A view and its owner read the same bytes. A write through either one is visible through the other, and `np.shares_memory` returns `True`.

**The failure.** A function that writes into its argument changes the data of the caller, as bug 1 showed. The cost is not a wrong number in the function. It is a wrong number somewhere else, later, with no link to the cause.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `root` | The function you build | the array that owns the memory of a view |
| `np.may_share_memory` | The function you build | quick test: do the memory ranges overlap |
| `np.ascontiguousarray` | What a copy costs, and what a bad walk costs | C-ordered array, copied only when needed |
| `np.ndarray.reshape` | Which operations copy | new shape, a view when the strides allow |
| `np.ndarray.copy` | Which operations copy | a new block, in a given memory order |

## What you have now

You have `root(a)`, and you can say which of your arrays share memory. You saw two bugs that come from views. A function changed its argument. A small view kept 288 MB alive.

You also measured the cost of a walk against the memory order. The asserts prove two bounds. A copy of the transpose is at least 1.3 times slower. A Python loop over columns is at least 3 times slower. The built-in reductions do not have this fault.

Chapter 4 uses the other half of the rule. Indexing with lists and masks makes copies, and writing through them has its own trap.
