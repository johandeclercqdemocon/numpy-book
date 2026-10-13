# Appendix B — What went wrong while writing

This list is the record of statements that the author wrote, and that a run or an audit then showed to be
wrong. They are fixed in the text. They are listed here because the same habits come back.

| Chapter | The wrong statement | What the run or the audit showed | How it was caught |
|---|---|---|---|
| all | each parameter table explains the parameters of its signature | 54 parameters of 23 functions had no row. On the first pages, `np.array` showed `subok`, `ndmax` and `like` in its signature and had no row for them. The checker tested that each row was a real parameter, and not that each parameter had a row | the reader, on page 4 of the PDF. The checker now fails for a parameter without a row, and `*args` and `**kwargs` count as parameters |
| [5](../chapters/05-broadcasting.md) | `np.broadcast_shapes` takes two or more shapes | it takes any number. One shape gives that shape, and no shape gives `()` | a run made while the missing rows were written |
| [1](../chapters/01-the-array.md) | in a draft row, `axis` of `np.linspace` has no effect when `start` and `stop` are numbers | `axis=1` then raises `AxisError` | running the call before the row went into the chapter |
| [1](../chapters/01-the-array.md) | the first example is a good first example: `np.arange(12, dtype=np.int32).reshape(3, 4)` | it needs two calls that the reader has not met, it makes a view of a block that no name points to, its values equal their positions, and it relies on a dtype that is not the default | a review by the reader, then `scripts/audit_examples.py` over every sample |
| [3](../chapters/03-views-and-copies.md) | `a.T.copy()` has strides `(16, 8)` and `np.copy(a.T)` has `(8, 24)` "for the array in the next block" | those strides belonged to another array. For the array of the block they are `(24, 8)` and `(8, 32)` | the review of every sample: the claim had no block that printed it |
| [6](../chapters/06-ufuncs-and-vectorization.md) | `np.add.reduce` gives `[3 5 7]` and `np.sum` gives `15` on a 2-D array | true for the array that the claim was made on, but no block printed the two calls | the same review. The block now prints both |
| [1](../chapters/01-the-array.md) | an array made with `reshape` owns its block | `owns=False`: it is a view of the array that `arange` made | the first output of `describe` |
| [2](../chapters/02-dtypes.md) | the `float32` cumulative sum is 8.8 percent too small | it is 8.8 percent too large | reading the printed total, 1087937 against 1000000 |
| [2](../chapters/02-dtypes.md) | each add of 0.1 is "below the gap and partly lost" | the add is rounded in both directions: -6 percent at one total, +25 percent at another | a block that measures one add at five totals |
| [3](../chapters/03-views-and-copies.md) | a view of a view points to the array one step up | NumPy points `.base` straight to the owner | printing `.base` of a view of a view |
| [3](../chapters/03-views-and-copies.md) | the quick test never says no when the answer is yes | untested at first. It holds on 5000 random slice pairs: 0 cases of the other kind | a block that counts both kinds of error |
| [5](../chapters/05-broadcasting.md) | the mean distance of the points is 6.6e-3 and the centered error is 8e-11 | 6.9e-3 and 1.2e-10 | the printed output |
| [5](../chapters/05-broadcasting.md) | the gap of neighbor `float32` values "is as large as the whole cube" | the gap is one tenth of the side of the cube | arithmetic, then the printed norms |
| [6](../chapters/06-ufuncs-and-vectorization.md) | "about 90 ufuncs" | 106 in the `numpy` namespace | counting them in a block |
| [6](../chapters/06-ufuncs-and-vectorization.md) | the centered cumulative sum has an error of about 1e-4 on seed 3 | 4.7e-4. On seed 2 it is exactly 0, and the reason is the size of the total | the printed output and the largest centered total |
| [6](../chapters/06-ufuncs-and-vectorization.md) | "there is no accumulate form" of the exponential moving average | a form with cumulative sums and powers exists | reading the claim as an enemy |
| [7](../chapters/07-linear-algebra-and-random.md) | the normal equations lose 8 digits at degree 9 | 7 digits: 1.3e-3 against 5.4e-11 | the printed output |
| [7](../chapters/07-linear-algebra-and-random.md) | `inv` has a residual 1e8 times larger at `n = 8` | 6e7 times | the printed output |
| [7](../chapters/07-linear-algebra-and-random.md) | `solve` takes 0.18 s and `inv` 0.48 s for 2000 | the ratio changes with the load of the machine and with the number of threads, down to 0.88 | the number audit, and a run on a busy machine |
| [8](../chapters/08-putting-it-together.md) | the first k-means run finds the three blobs | purity 0.67: a local minimum, on seed 0 | the assert of the first draft failed |
| [8](../chapters/08-putting-it-together.md) | the guessed timings 160 ms and 4.6 ms | 29 ms and 0.8 ms | running the block |
| all | timings typed into `# output varies` blocks | they came from other scripts. They are now printed by the block itself | `scripts/refresh_varies.py` |
| all | a timing assert passed when blocks ran in parallel and failed under load | the checker now runs timed blocks alone | a failed run, then the checker rule |

## Checks that found nothing

Four checks found no error in this book at the last run.

- The link check.
- The signature check of every `####` heading.
- The check that every `np.*` name in the text exists.
- The check that the defaults in the parameter tables are the defaults of the installed NumPy.
