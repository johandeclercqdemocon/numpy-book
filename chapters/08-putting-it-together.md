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


def kmeans(X, k, seed=0, max_iter=100):
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

The two versions agree on every label, and the largest centroid difference is 0.0 on this input. On other data the two versions can add in a different order and differ at the level of rounding. This is why the check uses `np.allclose` for the centroids and not `==`. If one label differs, the cause is almost always a tie that rounding broke the other way. In that case, compare the inertia.

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
300,000 points, vectorized: 319 ms
```

On 900 points the vectorized version is about 36 times faster on the build machine, and the assert requires 20. The loop time scales with the number of points. At 300,000 points it is 333 times larger, which is about 10 seconds, and the run does not include it.

This is the usual pattern. The check runs on a small input, and the fast version runs on the real one.

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

## What you have now

You have a clustering routine of 20 lines that you checked against a slow version, and a wrapper that fixes its one measured failure. You have a table of 13 traps, each with a measurement in an earlier chapter.

The book made one claim, and every chapter tested it: an array program is fast because of how it uses memory, and it is wrong in the places where it hides that use. The shape, the dtype and the strides from chapter 1 explain every row of the table above.
