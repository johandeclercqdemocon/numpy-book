# Chapter 8 — Putting it together

> **Build** — `kmeans`, a clustering routine of 20 lines with no Python loop over points, a loop version to check it, and a best-of-10 wrapper that fixes the one failure that the check finds.
>
> **Theory kept** — a vectorized program is a chain of the parts from earlier chapters, and each part has a cost in time, memory or precision. You test the chain against a slow, plain version of the same program.
>
> **Source** — New. The algorithm is Lloyd's k-means, which is standard. Every output below is from this book's code.
>
> **Run this first** — `uv run python -m examples.ch08_kmeans`, which is [`examples/ch08_kmeans.py`](../examples/ch08_kmeans.py).

The earlier chapters each gave one tool. This chapter uses four of them in one program and checks the result. The program is k-means clustering. It is small enough to read on one page. It needs a distance matrix (chapter 5), a group-by sum (chapter 4), a random generator that you pass (chapter 7) and the checks from chapters 2 and 3.

## The algorithm in two lines

Start with `k` centroids. Repeat two steps until the labels stop changing. First, give each point the label of its nearest centroid. Second, move each centroid to the mean of the points that have its label.

The first step is a distance matrix and an `argmin`. The second step is a group-by sum divided by a count. Both are in the book already.

## The function you build

```python
import numpy as np

from examples.ch05_broadcasting import dist_dot_centered      # chapter 5


def kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    """Lloyd's algorithm. Returns (centroids, labels, iterations)."""
    g = np.random.default_rng(seed)                           # chapter 7
    C = X[g.choice(len(X), k, replace=False)].copy()          # chapter 3: copy, not a view
    labels = np.full(len(X), -1)
    for it in range(1, max_iter + 1):
        new_labels = dist_dot_centered(X, C).argmin(axis=1)   # chapter 5
        if np.array_equal(new_labels, labels):
            return C, labels, it - 1
        labels = new_labels
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)                            # chapter 4
        counts = np.bincount(labels, minlength=k)
        keep = counts > 0                                     # an empty cluster keeps its centroid
        C[keep] = sums[keep] / counts[keep, None]             # chapter 5: broadcast a column
    return C, labels, max_iter


centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
g = np.random.default_rng(1)
X = np.concatenate([c + 0.8 * g.standard_normal((300, 2)) for c in centers])
truth = np.repeat(np.arange(3), 300)

C, labels, iters = kmeans(X, 3, seed=1)
print("iterations:", iters)
print("centroids:")
print(C.round(2))
print("cluster sizes:", np.bincount(labels))
```

```text
iterations: 3
centroids:
[[ 5.97 -0.06]
 [-0.07  0.  ]
 [ 3.02  5.02]]
cluster sizes: [300 300 300]
```

The three true centers are `(0, 0)`, `(6, 0)` and `(3, 5)`. The centroids are close to these, in another order. Each cluster has about 300 points. The `for` loop runs over iterations, and there are few of them. No loop runs over points.

#### `kmeans`

```text
kmeans(X: numpy.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[numpy.ndarray, numpy.ndarray, int]
```

The function runs Lloyd's algorithm. It returns the centroids, the label of each point and the number of iterations that changed a label.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `X` | required | the points, as an `(n, d)` array | a 1-D array raises an error in the distance step. Use `X.reshape(-1, 1)` for 1-D points |
| `k` | required | the number of clusters | `k` above `n` raises `ValueError` at the choice of starting points |
| `seed` | `0` | the seed of the choice of starting points | seed 0 stops at purity 0.67 on the data of this chapter, and 24 of 30 seeds find the blobs |
| `max_iter` | `100` | the most iterations | at the limit the function returns with labels that may still change, and the third value equals `max_iter` |

The function uses four more NumPy functions that need their signatures.

#### `np.random.Generator.choice`

```text
np.random.Generator.choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True)
```

The method picks elements of an array, or integers below an integer, at random.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | an array, or an integer `n` for the values `0 .. n-1` | — |
| `size` | `None` | how many to pick | — |
| `replace` | `True` | with `False`, no element is picked twice | with `True`, two starting centroids can be the same point |
| `p` | `None` | the probability of each element. With `None`, all are equal | — |
| `axis` | `0` | the axis of `a` to pick along, when `a` is an array. With `0`, the function picks rows | — |
| `shuffle` | `True` | with `replace=False`, puts the picked elements in random order. `False` is faster | with `False`, the order of the result is not random: `choice(10, 10, replace=False, shuffle=False)` gave `[0 1 2 ... 9]` |

#### `np.argmin`

```text
np.argmin(a, axis=None, out=None, *, keepdims=<no value>)
```

The function returns the index of the smallest value. If two values are equal, the first one wins.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array | — |
| `axis` | `None` | the axis to search along. With `None`, the flattened array | with `None` on an `(n, k)` matrix, you get one flat index and not one label for each point |
| `out` | `None` | an integer array to write the indices into. Its shape must be the shape of the result | a `float64` `out` raises `TypeError`, and a wrong shape raises `ValueError` |
| `keepdims` | `<no value>` | with `True`, the axis stays with length 1 | — |

#### `np.zeros_like`

```text
np.zeros_like(a, dtype=None, order='K', subok=True, shape=None, *, device=None)
```

The function returns an array of zeros with the shape of `a`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a` | required | the array whose shape is copied | — |
| `dtype` | `None` | the type of the result. With `None`, the type of `a` | an integer `a` gives an integer result, which cannot hold a mean |
| `order` | `'K'` | the memory order of the result. `'K'` keeps the order of `a` | — |
| `subok` | `True` | with `True`, a subclass of `ndarray`, such as a masked array, gives a result of that subclass | — |
| `shape` | `None` | a new shape for the result. With `None`, the shape of `a` | — |
| `device` | `None` | the device that holds the result. NumPy accepts only `'cpu'` | any other value raises `ValueError` |

#### `np.array_equal`

```text
np.array_equal(a1, a2, equal_nan=False)
```

The function returns `True` if two arrays have the same shape and the same values.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `a1`, `a2` | required | the two arrays | arrays of different shape give `False`, with no error |
| `equal_nan` | `False` | with `True`, two `nan` values count as equal | with `False`, an array that has a `nan` is never equal to itself |

The comments name the chapter of each line. The line `C[keep] = sums[keep] / counts[keep, None]` has a trap that chapter 5 described. The count has shape `(k,)` and the sums have shape `(k, d)`. Without `None`, the shapes `(k, d)` and `(k,)` do not align, or they align wrongly when `d` equals `k`.

## Check it against the slow version

A fast version that you did not check is a guess. The module has a second function, `kmeans_loop`, with the same algorithm and explicit Python loops, and the same starting centroids.

```python
import numpy as np

from examples.ch08_kmeans import kmeans, kmeans_loop, make_blobs

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, truth = make_blobs(300, centers, 0.8, seed=1)

C, labels, it = kmeans(X, 3, seed=1)
Cl, labels_l, it_l = kmeans_loop(X, 3, seed=1)
print("same labels:        ", np.array_equal(labels, labels_l))
print("same centroids:     ", np.allclose(C, Cl))
print("same iteration count:", it == it_l)
print("largest centroid difference:", np.abs(C - Cl).max())
```

```text
same labels:         True
same centroids:      True
same iteration count: True
largest centroid difference: 0.0
```

#### `make_blobs`

```text
make_blobs(n_per: int, centers: numpy.ndarray, spread: float, seed: int) -> tuple[numpy.ndarray, numpy.ndarray]
```

The function makes `n_per` points around each center, with Gaussian noise. It returns the points and the true label of each point.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `n_per` | required | the number of points for each center | — |
| `centers` | required | the centers, as a `(k, d)` array | — |
| `spread` | required | the standard deviation of the noise. The chapter uses 0.8 | with the centers 5 or 6 apart, a spread above about 2 makes the blobs overlap. The book did not measure the limit |
| `seed` | required | the seed of the noise | the same seed gives the same points |

#### `kmeans_loop`

```text
kmeans_loop(X: numpy.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[numpy.ndarray, numpy.ndarray, int]
```

The function is `kmeans` with explicit Python loops. It is the check, and it is not for use on real data.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `X` | required | the points, as an `(n, d)` array | slow: see the timing block, 900 points |
| `k` | required | the number of clusters | — |
| `seed` | `0` | the seed of the choice of starting points. The same seed gives the same start as `kmeans` | a different choice of start gives a different result, so compare at the same seed |
| `max_iter` | `100` | the most iterations | — |

The two versions agree on every label, and the largest centroid difference is 0.0 on this input. On other data the two versions can add in a different order and differ at the level of rounding. This is why the check uses `np.allclose` for the centroids and not `==`. If one label differs, a tie that rounding broke the other way is a possible cause. This book did not test such a case. Compare the inertia then.

## The check that found a real problem

Run the same function from 30 different seeds. The data are the same, and only the starting centroids differ. Count how often the run finds the three blobs, which we measure with *purity*: the fraction of points whose cluster has the same majority true label as they do.

```python
import numpy as np

from examples.ch08_kmeans import kmeans, kmeans_best, make_blobs, purity, inertia

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, truth = make_blobs(300, centers, 0.8, seed=1)

single = [purity(kmeans(X, 3, seed=s)[1], truth, 3) for s in range(30)]
good = sum(p > 0.95 for p in single)
print(f"single start: {good} of 30 seeds find the blobs")
print("purity of the failures:", sorted({round(float(p), 2) for p in single if p <= 0.95}))

inertias = {}
for s in range(30):
    C, labels, _ = kmeans(X, 3, seed=s)
    inertias[s] = (purity(labels, truth, 3) > 0.95, inertia(X, C, labels))
ok = [v for good, v in inertias.values() if good]
bad = [v for good, v in inertias.values() if not good]
print(f"inertia of the {len(ok)} good runs:    {min(ok):.1f} to {max(ok):.1f}")
print(f"inertia of the {len(bad)} failed runs:  {min(bad):.1f} to {max(bad):.1f}")

C, labels, _ = kmeans(X, 3, seed=0)
print("seed 0: rows are clusters, columns are true blobs")
print(np.array([np.bincount(truth[labels == j], minlength=3) for j in range(3)]))
```

```text
single start: 24 of 30 seeds find the blobs
purity of the failures: [0.67]
inertia of the 24 good runs:    1176.7 to 1176.7
inertia of the 6 failed runs:  6215.8 to 6503.7
seed 0: rows are clusters, columns are true blobs
[[  0 157   5]
 [  0 143   0]
 [300   0 295]]
```

In 6 of 30 runs, the algorithm stops at a purity of 0.67. The loop version agrees with the vectorized version on seed 1, so the cause is not a bug in the NumPy code. It is a property of the algorithm. The table for seed 0 shows the shape of the failure. One cluster holds all of blob 0 and nearly all of blob 2, so its centroid sits between them. The other two clusters split blob 1.

No single step improves this result. It is a local minimum.

Each run has an inertia, which is the sum of squared distances from the points to their centroids. The good runs all have the same inertia, and every failed run has a larger one. The inertia ranks the runs without the true labels, and this is why the standard fix works. Run the algorithm from several starts, and keep the run with the lowest inertia. The function `kmeans_best` does this with `SeedSequence.spawn`, as chapter 7 recommends.

```python
import numpy as np

from examples.ch08_kmeans import kmeans, kmeans_best, make_blobs, purity

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, truth = make_blobs(300, centers, 0.8, seed=1)

best = [purity(kmeans_best(X, 3, n_init=10, seed=s)[1], truth, 3) for s in range(30)]
print(f"best of 10 starts: {sum(p > 0.95 for p in best)} of 30 seeds find the blobs")
```

```text
best of 10 starts: 30 of 30 seeds find the blobs
```

#### `kmeans_best`

```text
kmeans_best(X: numpy.ndarray, k: int, n_init: int = 10, seed: int = 0) -> tuple[numpy.ndarray, numpy.ndarray, int]
```

The function runs `kmeans` from several starts and keeps the run with the lowest inertia.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `X` | required | the points, as an `(n, d)` array | the cost is `n_init` times the cost of `kmeans` |
| `k` | required | the number of clusters | — |
| `n_init` | `10` | the number of starts | with 1, it is `kmeans`, and 24 of 30 seeds succeed. With 10, 30 of 30 succeed |
| `seed` | `0` | the seed of the starts. The starts come from `SeedSequence.spawn` | — |

#### `inertia`

```text
inertia(X: numpy.ndarray, C: numpy.ndarray, labels: numpy.ndarray) -> float
```

The function returns the sum of squared distances from each point to its centroid.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `X` | required | the points | — |
| `C` | required | the centroids, as a `(k, d)` array | — |
| `labels` | required | the index of the centroid of each point | on the chapter data, the good runs have 1176.7 and the failed runs 6215.8 to 6503.7 |

#### `purity`

```text
purity(labels: numpy.ndarray, truth: numpy.ndarray, k: int) -> float
```

The function returns the fraction of points whose cluster has the same majority true label as they do.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `labels` | required | the cluster of each point | — |
| `truth` | required | the true label of each point, as non-negative integers | needs the true labels, so it cannot be used on real data |
| `k` | required | the number of clusters | — |

With 10 starts, all 30 seeds find the blobs. If one start fails with probability 0.2, the chance that all 10 fail is 0.2 to the power 10, which is 1e-7. The 30 results agree with that arithmetic.

## What the vectorization bought

```python
import timeit

import numpy as np

from examples.ch08_kmeans import kmeans, kmeans_loop, make_blobs

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
Xs, _ = make_blobs(300, centers, 0.8, seed=2)
t_loop = min(timeit.repeat(lambda: kmeans_loop(Xs, 3, seed=1), number=1, repeat=2))
t_vec = min(timeit.repeat(lambda: kmeans(Xs, 3, seed=1), number=1, repeat=5))
assert t_loop / t_vec > 20
print(f"900 points: loop {t_loop * 1e3:.0f} ms, vectorized {t_vec * 1e3:.1f} ms")

Xb, _ = make_blobs(100_000, centers, 0.8, seed=3)
t_big = min(timeit.repeat(lambda: kmeans(Xb, 3, seed=1), number=1, repeat=3))
print(f"300,000 points, vectorized: {t_big * 1e3:.0f} ms")
```

```text
# output varies
900 points: loop 29 ms, vectorized 0.8 ms
300,000 points, vectorized: 320 ms
```

On 900 points the assert requires the vectorized version to be at least 20 times faster. The loop time scales with the number of points. At 300,000 points it is 333 times larger. The run above printed 29 ms for 900 points, so the estimate is about 10 seconds, and the run does not include it.

This is the usual pattern. The check runs on a small input, and the fast version runs on the real one.

## How it works

Each iteration of `kmeans` is a chain of five steps, and every step has a shape. The next block runs one iteration by hand on the data of this chapter, and prints the shape of every value.

```python
import numpy as np

from examples.ch05_broadcasting import dist_dot_centered
from examples.ch08_kmeans import make_blobs

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, _ = make_blobs(300, centers, 0.8, seed=1)
k = 3
C = X[np.random.default_rng(1).choice(len(X), k, replace=False)].copy()

D = dist_dot_centered(X, C)
labels = D.argmin(axis=1)
sums = np.zeros_like(C)
np.add.at(sums, labels, X)
counts = np.bincount(labels, minlength=k)
new_C = sums / counts[:, None]

for name, value in [("X", X), ("C", C), ("D = distances", D), ("labels", labels),
                    ("sums", sums), ("counts", counts), ("counts[:, None]", counts[:, None]),
                    ("new centroids", new_C)]:
    print(f"{name:18s} {value.shape}")
print("values in D, the cost of the distance step:", D.size, "= n * k =", len(X) * k)
```

```text
X                  (900, 2)
C                  (3, 2)
D = distances      (900, 3)
labels             (900,)
sums               (3, 2)
counts             (3,)
counts[:, None]    (3, 1)
new centroids      (3, 2)
values in D, the cost of the distance step: 2700 = n * k = 2700
```

The distance step has the cost `n * k * d` multiplications, and the group sum has the cost `n * d` additions. Both are linear in the number of points. The loop over iterations has few steps, so the total cost is linear in `n`.

The reason that the algorithm works is one fact. Each of the two steps does not make the inertia larger. The label step gives each point the nearest centroid. The mean step puts each centroid at the point that minimizes the squared distances of its cluster. The next block records the inertia after every label step for 30 seeds, and tests that it never rises.

```python
import numpy as np

from examples.ch05_broadcasting import dist_dot_centered
from examples.ch08_kmeans import inertia, make_blobs

centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, _ = make_blobs(300, centers, 0.8, seed=1)
k = 3

never_rises, iterations = 0, []
for seed in range(30):
    C = X[np.random.default_rng(seed).choice(len(X), k, replace=False)].copy()
    history = []
    for _ in range(100):
        labels = dist_dot_centered(X, C).argmin(axis=1)
        history.append(inertia(X, C, labels))
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)
        counts = np.bincount(labels, minlength=k)
        keep = counts > 0
        C[keep] = sums[keep] / counts[keep, None]
        if len(history) > 1 and np.isclose(history[-1], history[-2], rtol=0, atol=1e-9):
            break
    never_rises += all(b <= a + 1e-9 for a, b in zip(history, history[1:]))
    iterations.append(len(history))

print("seeds where the inertia never rose:", never_rises, "of 30")
print("iterations until it stopped: least", min(iterations), "most", max(iterations))
```

```text
seeds where the inertia never rose: 30 of 30
iterations until it stopped: least 3 most 19
```

**The invariant.** The inertia never rises from one iteration to the next. The block above tests it on 30 starts.

**The failure.** The invariant says that the algorithm stops, and not that it stops at the best answer. It stops at the first point where no step helps, which is a local minimum. Six of the 30 seeds stop there. Their inertia is 5.3 to 5.5 times the best value, which is 1176.7.

## The traps, in the order you meet them

Each row is a mistake that gives a wrong answer and no error. The chapter column says where this book measured it.

| Trap | Why it happens | How to catch it | Chapter |
|---|---|---|---|
| An array has a different dtype than you assumed | NumPy picks the dtype from the data | Print `.dtype` at every input and output | 2 |
| Integers wrap | The dtype fixes the number of bits | `checked_add`, or a wider dtype | 2 |
| A `float32` running sum drifts | Small values are added to a large total | Use `x.sum()`, or `float64` | 2 |
| A function changes its argument | A slice is a view | `np.shares_memory`, `root` | 3 |
| A small view keeps a large block alive | The view holds a reference to its base | `.copy()` | 3 |
| A write is lost | A mask or index list makes a copy | Write with the mask on the left | 4 |
| A repeated index is applied once | `a[i] += 1` reads, adds, then writes | `np.add.at`, `np.bincount` | 4 |
| A result is `(n, n)` and not `(n,)` | Broadcasting stretches a length-1 axis | Assert the shape of a result | 5 |
| The distance matrix is wrong in `float32` | `|a|² + |b|² − 2a·b` subtracts large numbers | Center the data first | 5 |
| `np.vectorize` is no faster | It is a Python loop | Use ufuncs and `np.where` | 6 |
| `inv(A) @ b` is inaccurate | The inverse amplifies rounding | `solve`, and check the residual | 7 |
| A seed gives other numbers on another machine | The stream depends on the NumPy version | Write the version next to the seed | 7 |
| An iterative method stops at a poor answer | A local minimum | Several starts, keep the lowest inertia | 8 |

The most useful single habit is the test in this chapter. Write the slow version first. It is short, it is obviously right, and it costs nothing to keep. Then write the fast version, and assert that the two agree on a small input.

## What this book left out

This book is short, so it skips topics that have a real use. They are listed here so that you know where the edge is. This book did not measure any of them.

- `np.einsum` writes a sum over named axes in one call, and replaces many `@` and broadcast chains.
- Structured arrays and `np.recarray` hold records with named fields of mixed types.
- Masked arrays hold missing values with a mask and not a `nan`.
- `np.memmap` and `np.load(mmap_mode="r")` read an array from disk without loading it.
- `np.fft` computes Fourier transforms, and `np.linalg` has factorizations beyond those used here.
- Compilers such as Numba, and libraries such as SciPy, cover the loops that chapter 6 could not vectorize.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `kmeans` | The function you build | Lloyd's algorithm with no loop over points |
| `np.random.Generator.choice` | The function you build | random picks from an array, with or without replacement |
| `np.argmin` | The function you build | index of the smallest value along an axis |
| `np.zeros_like` | The function you build | zeros with the shape and type of another array |
| `np.array_equal` | The function you build | same shape and same values |
| `make_blobs` | Check it against the slow version | points around given centers, with the true labels |
| `kmeans_loop` | Check it against the slow version | the same algorithm with Python loops |
| `kmeans_best` | The check that found a real problem | the run with the lowest inertia of several starts |
| `inertia` | The check that found a real problem | sum of squared distances to the centroids |
| `purity` | The check that found a real problem | share of points in the majority cluster of their true label |

## What you have now

You have a clustering routine of 20 lines that you checked against a slow version, and a wrapper that fixes its one measured failure. You have a table of 13 traps, each with a measurement in an earlier chapter.

The book made one claim, and every chapter tested it: an array program is fast because of how it uses memory, and it is wrong in the places where it hides that use. The shape, the dtype and the strides from chapter 1 explain every row of the table above.
