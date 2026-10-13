# Appendix A — Described, not run

Every row below is a statement in the book that no block of the book tested. Each one is worded in the
text as a statement of the documentation, as a likely cause, or as a limit of the book. Nothing in this
list is a measured result.

| Chapter | Statement | Where it comes from | How you can test it |
|---|---|---|---|
| [1](../chapters/01-the-array.md) | NumPy 1.x made an array of objects from a list of unequal rows | memory of the NumPy 1.x behavior | install NumPy 1.26 and run `np.array([[1, 2], [3]])` |
| [1](../chapters/01-the-array.md) | a list holds pointers to separate integer objects, and a sum must follow each pointer | the CPython object model | read `sys.getsizeof` of the list and of one item, and time `sum` on a list of floats |
| [2](../chapters/02-dtypes.md) | NumPy 1.x chose the result dtype from the value of a Python number | the NumPy documentation of NEP 50 | install NumPy 1.26 and run `np.uint8(1) + 300` |
| [2](../chapters/02-dtypes.md) | `x.sum()` adds in pairs, and that is why it is accurate | the NumPy documentation of `np.sum` | add the values in a Python loop and in a pairwise tree, and compare both with the exact sum |
| [3](../chapters/03-views-and-copies.md) | a strided read is slow because it wastes the line of 64 bytes that a read brings in | the usual model of a CPU cache | read with strides of 1, 2, 4, 8, 16 and 32 values, and look for steps in the time |
| [3](../chapters/03-views-and-copies.md) | the result of `np.broadcast_to` is read-only because elements share a location | the NumPy documentation of `np.broadcast_to` | none: the read-only flag is printed in chapter 3 |
| [3](../chapters/03-views-and-copies.md) | the `OWNDATA` flag tells which array frees the block | the NumPy documentation of `ndarray.flags` | none that is simple |
| [4](../chapters/04-indexing.md) | `np.add.at` was slow in NumPy 1.x | memory of release notes | install NumPy 1.26 and time the block of chapter 4 |
| [5](../chapters/05-broadcasting.md) | the dot version is fast because it never forms the `(n, m, d)` array | a likely cause | time a version that forms the array and one that does not, with the same arithmetic |
| [6](../chapters/06-ufuncs-and-vectorization.md) | a ufunc picks the first loop to which both inputs can be cast safely, and casts in small buffers | the NumPy documentation of ufuncs | `np.add.resolve_dtypes`, and the `bufsize` of `np.nditer` |
| [6](../chapters/06-ufuncs-and-vectorization.md) | the window version is slower because it takes a real mean over each row | a likely cause | count the additions of the window version and of the cumulative-sum version |
| [6](../chapters/06-ufuncs-and-vectorization.md) | an exponential moving average can be written with cumulative sums and powers | the algebra of the recurrence | write it, and compare it with the loop on 1e6 values |
| [6](../chapters/06-ufuncs-and-vectorization.md) | a filter routine and Numba are faster than the loop | memory | time both on the loop of chapter 6 |
| [7](../chapters/07-linear-algebra-and-random.md) | the loss of digits of the normal equations is the squared condition number | the theory of least squares | the column `cond(XtX)` in the table of chapter 7 is consistent with it. Vary the degree finely and fit the slope |
| [7](../chapters/07-linear-algebra-and-random.md) | the Hilbert matrix loses the same digits in `solve` and in `lu_solve` because of its condition number | a likely cause | compare both on a well-conditioned matrix of the same size |
| [7](../chapters/07-linear-algebra-and-random.md) | the streams of `Generator` methods can change between NumPy versions | the NumPy policy for random numbers | install two versions and compare the first values for one seed |
| [7](../chapters/07-linear-algebra-and-random.md) | `spawn_key` goes into the start state | the NumPy documentation of `SeedSequence` | the block of chapter 7 shows that the start states differ, not why |
| [8](../chapters/08-putting-it-together.md) | a label that differs between `kmeans` and `kmeans_loop` is a tie that rounding broke | a possible cause | build a data set with exact ties and compare |
| [8](../chapters/08-putting-it-together.md) | blobs overlap when the spread is above about 2 | a rough guess, not measured | run `kmeans_best` for spreads from 0.5 to 4 and plot the purity |
| [8](../chapters/08-putting-it-together.md) | k-means++ needs fewer restarts | memory | implement it and count the failures on the data of chapter 8 |
