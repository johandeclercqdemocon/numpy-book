# Chapter 4 — Indexing

> **Build** — `group_sum(keys, values, k)`, which adds up the values for each key, in three versions that you can time against each other.
>
> **Theory kept** — an index with integers or a mask picks elements and makes a copy. A write with an index list applies to each position once. A condition in `np.where` is computed on both sides.
>
> **Source** — New. The behavior is from the NumPy documentation on advanced indexing. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch04_indexing`, which is [`examples/ch04_indexing.py`](../examples/ch04_indexing.py).

Chapter 3 said that a list of integers or a mask makes a copy. This chapter shows the four traps that follow from that rule. Each trap produces a wrong result with no error. The chapter then builds a function that counts by key, which is the most common reason for an index with repeated values.

## Reading with a mask

A mask is a boolean array of the same shape as the data. A comparison makes one. The mask can count, select and replace.

```python
import numpy as np

x = np.array([3, -1, 4, -1, 5, -9, 2, 6])
mask = x < 0
print("mask:    ", mask)
print("count:   ", mask.sum())
print("selected:", x[mask])
print("fraction:", mask.mean())
```

```text
mask:     [False  True False  True False  True False False]
count:    3
selected: [-1 -1 -9]
fraction: 0.375
```

The sum of a boolean array counts the `True` values. The mean gives the fraction. Both use the fact that `True` is 1.

## Trap 1: a write to a copy is lost

A mask on the left of `=` writes into the array. A mask on the right makes a copy, and a write into the copy is lost.

```python
import numpy as np

m = np.arange(10)
m[m > 5][0] = 99
print("chained write:", m)

m[m > 5] = 99
print("direct write: ", m)
```

```text
chained write: [0 1 2 3 4 5 6 7 8 9]
direct write:  [ 0  1  2  3  4  5 99 99 99 99]
```

The first line looks like it sets one element to 99. Python runs two steps: `m[m > 5]` makes a copy, and `[0] = 99` writes into that copy. The copy is then deleted. The array `m` did not change, and NumPy did not say so. The second line is one step, and it works.

## Trap 2: a repeated index is applied once

Add 1 at each position in an index list. If one position appears three times, you want 3. You get 1.

```python
import numpy as np

idx = np.array([1, 1, 3, 1])

a = np.zeros(5, dtype=int)
a[idx] += 1
print("a[idx] += 1  :", a)

b = np.zeros(5, dtype=int)
np.add.at(b, idx, 1)
print("np.add.at    :", b)

print("np.bincount  :", np.bincount(idx, minlength=5))
```

```text
a[idx] += 1  : [0 1 0 1 0]
np.add.at    : [0 3 0 1 0]
np.bincount  : [0 3 0 1 0]
```

#### `np.add.at`

```text
np.add.at(a, indices, b=None, /)
```

The method adds `b` to `a` at each position in `indices`, one element at a time. It changes `a` in place and returns `None`.

The table under `np.add.at` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `np.bincount`

```text
np.bincount(x, /, weights=None, minlength=0)
```

The function counts how many times each integer occurs, or adds a weight for each occurrence.

The table under `np.bincount` in the Reference section of this chapter gives each parameter and what goes wrong with it.

The statement `a[idx] += 1` is `a[idx] = a[idx] + 1`. NumPy reads `a[idx]` once, adds 1 to each of the four values, and writes four results. Position 1 receives the same result three times. The function `np.add.at` goes one element at a time, so it is correct. The function `np.bincount` counts directly.

## Trap 3: `np.where` computes both sides

#### `np.where`

```text
np.where(condition, x=None, y=None, /)
```

With three arguments, the function picks from `x` where the condition is `True` and from `y` where it is `False`. With one argument, it returns the positions where the condition is `True`.

The table under `np.where` in the Reference section of this chapter gives each parameter and what goes wrong with it.

`np.where(cond, x, y)` picks values from `x` or from `y`. Both `x` and `y` are computed first. A protection that you put in the condition does not stop a bad value in the branch.

```python
import warnings

import numpy as np

x = np.array([4.0, 0.0, 2.0])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = np.where(x != 0, 1 / x, 0.0)
print("result:  ", r)
print("warnings:", [str(w.message) for w in caught])

r2 = np.divide(1, x, out=np.zeros_like(x), where=x != 0)
print("with where=:", r2)
```

```text
result:   [0.25 0.   0.5 ]
warnings: ['divide by zero encountered in divide']
with where=: [0.25 0.   0.5 ]
```

The result is right and the warning shows that the division by zero ran. For division the harm is small, because the bad value is not selected. For `np.log` of a negative number, the branch that you did not select gives a warning. For an index that is out of range, it raises an error. The `where=` argument of a ufunc skips the elements where the condition is false. Chapter 6 covers this argument.

## Trap 4: the shape of a mixed index

When you mix a slice and an index list in one index, NumPy can move the axes of the result.

```python
import numpy as np

A = np.zeros((2, 3, 4))      # only the shapes matter here
print(A[0, :, [0, 1]].shape)
print(A[:, [0, 1], [0, 1]].shape)
print(A[0][:, [0, 1]].shape)
```

```text
(2, 3)
(2, 2)
(3, 2)
```

Read the first line. The index picks `A[0]` and then columns 0 and 1, and you can expect the shape `(3, 2)`. The result is `(2, 3)`. The rule is that the axes that come from index lists go first when a slice separates two advanced indices. In the first line the scalar `0` and the list `[0, 1]` are both advanced indices, and the slice sits between them.

The third line avoids the problem. It indexes in two steps, and the shape is `(3, 2)`. When the shape of a mixed index matters, use two steps or print `.shape` before you rely on it.

## The function you build

A group-by sum is the most common use of a repeated index. You have one value per row and one key per row, and you want one total per key. Three versions follow. They must give the same answer.

```python
import numpy as np


def group_sum_loop(keys: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Add up `values` for each key in 0..k-1 with a Python loop."""
    out = [0.0] * k
    for key, v in zip(keys.tolist(), values.tolist()):
        out[key] += v
    return np.array(out)


def group_sum_at(keys: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Add up `values` for each key in 0..k-1 with np.add.at."""
    out = np.zeros(k)
    np.add.at(out, keys, values)
    return out


def group_sum_bincount(keys: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Add up `values` for each key in 0..k-1 with np.bincount."""
    return np.bincount(keys, weights=values, minlength=k)


rng = np.random.default_rng(0)
n, k = 1_000_000, 100
keys = rng.integers(0, k, n)
values = rng.random(n)

ref = group_sum_bincount(keys, values, k)
print("loop    matches bincount:", np.allclose(group_sum_loop(keys, values, k), ref))
print("add.at  matches bincount:", np.allclose(group_sum_at(keys, values, k), ref))
print("total of all groups     :", round(float(ref.sum()), 3), "=", round(float(values.sum()), 3))
```

```text
loop    matches bincount: True
add.at  matches bincount: True
total of all groups     : 500316.548 = 500316.548
```

The third line is a second check that does not depend on the other functions. The total of the groups must equal the total of all values.

Now time the three versions.

```python
import timeit

import numpy as np

rng = np.random.default_rng(0)
n, k = 1_000_000, 100
keys = rng.integers(0, k, n)
values = rng.random(n)


def loop():
    out = [0.0] * k
    for key, v in zip(keys.tolist(), values.tolist()):
        out[key] += v


def at():
    out = np.zeros(k)
    np.add.at(out, keys, values)


def bincount():
    np.bincount(keys, weights=values, minlength=k)


t = {f.__name__: min(timeit.repeat(f, number=1, repeat=3)) for f in (loop, at, bincount)}
assert t["loop"] / t["at"] > 5
assert t["loop"] / t["bincount"] > 5
assert 0.5 < t["at"] / t["bincount"] < 2          # the two NumPy versions are within a factor of 2
for name, sec in t.items():
    print(f"{name:9s} {sec * 1e3:8.2f} ms")
```

```text
# output varies
loop        111.35 ms
at            2.53 ms
bincount      2.69 ms
```

The assert proves that the loop is more than 5 times slower than either NumPy version. That is the expected result. The surprise is the second line against the third line.

Old advice says that `np.add.at` is very slow and that you must use `np.bincount`. That advice dates from NumPy 1.x, which this book did not install, so it cannot say whether it was true there. On NumPy 2.5.4 the assert proves that the two are within a factor of 2 of each other on this input.

Check the version of NumPy before you accept advice about speed. The `bincount` version has one restriction. The keys must be non-negative integers. The function `np.add.at` accepts any index that is valid for the output array.

#### `group_sum_loop`

```text
group_sum_loop(keys: numpy.ndarray, values: numpy.ndarray, k: int) -> numpy.ndarray
```

The function adds the values of each group with a Python loop. It is the definition, and the other two are checked against it.

The table under `group_sum_loop` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `group_sum_at`

```text
group_sum_at(keys: numpy.ndarray, values: numpy.ndarray, k: int) -> numpy.ndarray
```

The function adds the values with `np.add.at`, which handles a repeated key correctly.

The table under `group_sum_at` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `group_sum_bincount`

```text
group_sum_bincount(keys: numpy.ndarray, values: numpy.ndarray, k: int) -> numpy.ndarray
```

The function adds the values with `np.bincount`, which accepts only non-negative integer keys.

The table under `group_sum_bincount` in the Reference section of this chapter gives each parameter and what goes wrong with it.

## Finding the largest values without a sort

A full sort puts every value in order. To find the 10 largest of 5 million values, you do not need that. `np.partition` puts the 10 largest at the end, in any order, and it does less work.

```python
import timeit

import numpy as np

x = np.random.default_rng(1).random(5_000_000)
top_sort = np.sort(x)[-10:]
top_part = np.sort(np.partition(x, -10)[-10:])
print("same ten values:", np.array_equal(top_sort, top_part))

t_sort = min(timeit.repeat(lambda: np.sort(x)[-10:], number=1, repeat=3))
t_part = min(timeit.repeat(lambda: np.partition(x, -10)[-10:], number=1, repeat=3))
assert t_sort / t_part > 2
print(f"sort / partition: {t_sort / t_part:.1f}x")
```

```text
# output varies
same ten values: True
sort / partition: 4.8x
```

The partition result is not in order, so the code sorts the 10 values at the end. The assert requires a saving of at least 2 times.

#### `np.partition`

```text
np.partition(a, kth, axis=-1, kind='introselect', order=None)
```

The function puts the `kth` smallest value in its sorted position. It puts smaller values before it and larger values after it, in no order.

The table under `np.partition` in the Reference section of this chapter gives each parameter and what goes wrong with it.

## How it works

An index with integer arrays or a mask is a gather. NumPy broadcasts the index arrays against each other. It reads one element for each position of the broadcast shape. It writes the elements into a new array. This explains the four traps, and the next blocks test each part.

First, a mask is a list of integer indices in disguise. NumPy turns the mask into the positions of its `True` values, then gathers.

```python
import numpy as np

rng = np.random.default_rng(0)
m = rng.random((4, 5))
mask = m > 0.5
positions = np.nonzero(mask)

print("mask picks the same values as the positions:", np.array_equal(m[mask], m[positions]))
print("positions are two arrays, one for each axis:", [p.shape for p in positions])
```

```text
mask picks the same values as the positions: True
positions are two arrays, one for each axis: [(12,), (12,)]
```

Second, the shape of the result is the broadcast shape of the index arrays. A column of row numbers and a row of column numbers give a grid, as in chapter 5. The array `m` below is built with `np.arange(20).reshape(4, 5)` on purpose. The value of each cell is its position in the flat order, so the output shows which cells the index picked.

```python
import numpy as np

m = np.arange(20).reshape(4, 5)
rows = np.array([[0], [2]])
cols = np.array([1, 3, 4])

picked = m[rows, cols]
print("index shapes:", rows.shape, cols.shape)
print("broadcast shape:", np.broadcast_shapes(rows.shape, cols.shape))
print("result shape:", picked.shape)
print(picked)
```

```text
index shapes: (2, 1) (3,)
broadcast shape: (2, 3)
result shape: (2, 3)
[[ 1  3  4]
 [11 13 14]]
```

Third, a write is a scatter. If a position appears more than once, the last value stays, and an in-place add reads the old value only once. The next block spells out `a[idx] += 1` in three steps.

```python
import numpy as np

v = np.array([0, 10, 20, 30, 40])
idx = np.array([1, 1, 3])

fast = v.copy()
fast[idx] += 1

slow = v.copy()
old = slow[idx]          # step 1: gather. Position 1 is read twice, and both reads give 10
new = old + 1            # step 2: add. Both copies become 11
slow[idx] = new          # step 3: scatter. Position 1 is written twice with 11

print("same as the one-line form:", np.array_equal(fast, slow))
print("result:", fast, "and position 1 went up by 1, not 2")

last = np.zeros(2)
last[[0, 0]] = [1.0, 2.0]
print("repeated index in a plain write, the last value stays:", last)
```

```text
same as the one-line form: True
result: [ 0 11 20 31 40] and position 1 went up by 1, not 2
repeated index in a plain write, the last value stays: [2. 0.]
```

Fourth, `np.take` is the gather without the broadcast rule, and it has a `mode` for positions out of range. The indexing form raises `IndexError` for a position out of range. The method `np.take` can clip.

```python
import numpy as np

x = np.arange(3)
try:
    x[[5]]
except IndexError as e:
    print("x[[5]]:", e)
print("take, mode='clip':", x.take([5], mode="clip"))
print("take, mode='wrap':", x.take([5], mode="wrap"))
```

```text
x[[5]]: index 5 is out of bounds for axis 0 with size 3
take, mode='clip': [2]
take, mode='wrap': [2]
```

The position 5 is clipped to the last element, 2, or wrapped to `5 % 3`, which is 2 as well. Both modes hide an index that is wrong. Use them only when that is what you mean.

**The invariant.** The result of an index with arrays has the broadcast shape of the index arrays, and it is a new block. The result of a basic slice is a view.

**The failure.** The write `a[idx] += 1` is a gather, an add and a scatter. A repeated index loses all but one of its updates. The block above shows position 1 going up by 1. The loop `for i in idx: a[i] += 1` adds 2.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `np.add.at` | Trap 2: a repeated index is applied once | add at positions, one at a time, in place |
| `np.where` | Trap 3: `np.where` computes both sides | pick from two arrays by a condition |
| `np.bincount` | Trap 2: a repeated index is applied once | count or sum weights for each non-negative integer |
| `group_sum_loop` | The function you build | group sum, the definition |
| `group_sum_at` | The function you build | group sum with `np.add.at` |
| `group_sum_bincount` | The function you build | group sum with `np.bincount` |
| `np.partition` | Finding the largest values without a sort | the k-th value in place, the rest unordered |

### Parameters

##### `np.add.at`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array to change. It is changed in place | a result that you expect from the call does not exist: the return value is `None` |
| `indices` | required | an index list, or a tuple of index lists for several axes. A position can appear many times | a negative index counts from the end, so `-1` changes the last element |
| `b` | `None` | the value to add. A scalar or an array that broadcasts to the indexed shape | — |

##### `np.bincount`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | a 1-D array of non-negative integers | a negative value raises `ValueError` |
| `weights` | `None` | one weight for each element of `x`. The result then holds the sum of the weights, as `float64` | the weights make the result `float64`, also when they are integers |
| `minlength` | `0` | the least length of the result | without it, the result has length `max(x) + 1`, and it can differ between two runs on different data |

##### `np.where`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `condition` | required | a boolean array, or an array that NumPy reads as boolean | a count of 0 and 1 values is read as a condition: `0` is `False` and every other value is `True` |
| `x` | `None` | the values where the condition is `True` | it is computed for every element, also the ones that the condition rejects |
| `y` | `None` | the values where the condition is `False` | it is computed for every element, also the ones that the condition accepts |

##### `group_sum_loop`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `keys` | required | one non-negative integer for each row, in `0 .. k-1` | a key of `k` or more raises `IndexError` |
| `values` | required | one number for each row, with the same length as `keys` | `zip` stops at the shorter input, so a length mismatch gives no error |
| `k` | required | the number of groups, which is the length of the result | — |

##### `group_sum_at`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `keys` | required | one integer for each row, in `0 .. k-1` | a key of `k` or more raises `IndexError`. A key of `-1` adds to the last group |
| `values` | required | one number for each row, with the same length as `keys` | a different length raises an error |
| `k` | required | the number of groups, which is the length of the result | — |

##### `group_sum_bincount`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `keys` | required | one non-negative integer for each row | a negative key raises `ValueError` |
| `values` | required | one number for each row, with the same length as `keys` | a different length raises an error |
| `k` | required | the least length of the result | a key of `k` or more makes the result longer than `k`, with no error |

##### `np.partition`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array to partition. The function returns a copy | the input is not changed, and the result is a new block |
| `kth` | required | the index of the value to put in place. A negative index counts from the end, so `-10` puts the 10 largest at the end | the values on each side are not sorted |
| `axis` | `-1` | the axis to partition along | — |
| `kind` | `'introselect'` | the selection algorithm. NumPy has one | — |
| `order` | `None` | for an array with named fields (a structured array), the field names to compare | on an array without fields, any value raises `ValueError` |


## What you have now

You have three versions of `group_sum` that give the same answer, and you know which one to use first: `np.bincount` when the keys are small non-negative integers, and `np.add.at` for the other cases. You know four traps by name.

- A chained write is lost.
- A repeated index is applied once.
- The `np.where` function computes both sides.
- A mixed index moves axes.

Chapter 5 uses indexing in the other direction. It does not pick elements from an array. It stretches arrays so that they combine without any loop.
