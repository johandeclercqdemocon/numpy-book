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

The statement `a[idx] += 1` is `a[idx] = a[idx] + 1`. NumPy reads `a[idx]` once, adds 1 to each of the four values, and writes four results. Position 1 receives the same result three times. The function `np.add.at` goes one element at a time, so it is correct. The function `np.bincount` counts directly.

## Trap 3: `np.where` computes both sides

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

A = np.arange(24).reshape(2, 3, 4)
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


def group_sum_loop(keys, values, k):
    out = [0.0] * k
    for key, v in zip(keys.tolist(), values.tolist()):
        out[key] += v
    return np.array(out)


def group_sum_at(keys, values, k):
    out = np.zeros(k)
    np.add.at(out, keys, values)
    return out


def group_sum_bincount(keys, values, k):
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
for name, sec in t.items():
    print(f"{name:9s} {sec * 1e3:8.2f} ms")
```

```text
# output varies
loop        113.62 ms
at            2.74 ms
bincount      2.23 ms
```

The loop is 40 to 50 times slower than either NumPy version. That is the expected result. The surprise is the second line against the third line.

Old advice says that `np.add.at` is very slow and that you must use `np.bincount`. That advice was true for NumPy 1.x. On NumPy 2.5.4 the two are within 25 percent of each other on this input.

Check the version of NumPy before you accept advice about speed. The `bincount` version has one restriction. The keys must be non-negative integers. The function `np.add.at` accepts any index that is valid for the output array.

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
sort / partition: 4.7x
```

The partition result is not in order, so the code sorts the 10 values at the end. The saving is a factor of about 5 on the build machine. The assert requires at least 2.

## What you have now

You have three versions of `group_sum` that give the same answer, and you know which one to use first: `np.bincount` when the keys are small non-negative integers, and `np.add.at` for the other cases. You know four traps by name.

- A chained write is lost.
- A repeated index is applied once.
- The `np.where` function computes both sides.
- A mixed index moves axes.

Chapter 5 uses indexing in the other direction. It does not pick elements from an array. It stretches arrays so that they combine without any loop.
