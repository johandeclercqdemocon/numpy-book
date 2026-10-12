# NumPy, Measured

*Arrays, dtypes, views, broadcasting, and the mistakes that give a wrong answer and no error.*

This is a short book about NumPy for a reader who already writes Python. It has eight chapters. Each chapter ends with something that you built and tested. The book keeps theory to the minimum that changes what you type. Its subject is one idea: an array is a block of memory plus a description of how to read it, and every surprise in NumPy comes from that description.

**Nothing in it is asserted that was not run.** Every number in the text comes from code that ships with the book. A checker executes every code block and compares every printed output with the text. The book depends only on NumPy, and the whole book verifies offline and costs nothing: no network, no model, no API.

The text is in Simplified Technical English. NumPy 2.5.4 on Python 3.13 produced every output in the book.

## Contents

| # | Chapter | You build | Run first |
|---|---|---|---|
| 1 | [The array](chapters/01-the-array.md) | `describe(a)`, a one-line memory layout, and a measured list-versus-array sum | `uv run python -m examples.ch01_array` |
| 2 | [Dtypes, and the answers that change silently](chapters/02-dtypes.md) | `checked_add`, an integer add that refuses to wrap | `uv run python -m examples.ch02_dtypes` |
| 3 | [Views and copies](chapters/03-views-and-copies.md) | `root(a)`, and a table of which operations copy | `uv run python -m examples.ch03_views` |
| 4 | [Indexing](chapters/04-indexing.md) | `group_sum` in three versions | `uv run python -m examples.ch04_indexing` |
| 5 | [Broadcasting](chapters/05-broadcasting.md) | pairwise distances: loop, broadcast and dot, with time, memory and precision | `uv run python -m examples.ch05_broadcasting` |
| 6 | [Ufuncs and vectorization](chapters/06-ufuncs-and-vectorization.md) | a moving average in four versions | `uv run python -m examples.ch06_ufuncs` |
| 7 | [Linear algebra and random numbers](chapters/07-linear-algebra-and-random.md) | `fit_line` with a reproducible bootstrap interval | `uv run python -m examples.ch07_linalg_random` |
| 8 | [Putting it together](chapters/08-putting-it-together.md) | `kmeans`, checked against a loop version, and a table of 13 traps | `uv run python -m examples.ch08_kmeans` |

Each chapter has three exercises, one of each kind, and a worked solution.

| # | Exercises | Solutions |
|---|---|---|
| 1 | [01](exercises/01.md) | [01](exercises/solutions/01.md) |
| 2 | [02](exercises/02.md) | [02](exercises/solutions/02.md) |
| 3 | [03](exercises/03.md) | [03](exercises/solutions/03.md) |
| 4 | [04](exercises/04.md) | [04](exercises/solutions/04.md) |
| 5 | [05](exercises/05.md) | [05](exercises/solutions/05.md) |
| 6 | [06](exercises/06.md) | [06](exercises/solutions/06.md) |
| 7 | [07](exercises/07.md) | [07](exercises/solutions/07.md) |
| 8 | [08](exercises/08.md) | [08](exercises/solutions/08.md) |

The three kinds are as follows.

- **[measure]** asks for a number with a tolerance.
- **[break]** changes one thing so that a result reverses.
- **[judge]** has no single right answer.

A judge solution gives a position, the strongest objection and the evidence that overturns it.

## How to run it

```bash
cd ~/claude/numpy-book
uv sync
uv run python scripts/check_snippets.py
```

The checker verifies four things:

- Every relative link resolves to a file that exists.
- Every chapter has its `Build`, `Theory kept`, `Source` and `Run this first` header.
- Every `python` block in a chapter, an exercise or a solution runs and exits with 0.
- Every `text` block under a `python` block equals the real output.

A block that measures time runs alone, after the others, so that the asserted ratios are not distorted. Run the checker on an idle machine.

## Where the book is honest about its limits

- **Timings are from one machine.** A block that prints a time starts its output with `# output varies`. The block has an `assert` with the claim, for example "at least 5 times faster", and the assert is what the checker enforces. The printed numbers are from the build machine. Yours will differ.
- **Some results are properties of the data.** In chapter 6 the centered cumulative sum has an error of exactly 0 on one seed. The chapter says that this is luck of the data, and it shows a second seed.
- **The chapters are short.** They have 1,300 to 2,200 words each, with code. A longer book can cover `einsum`, structured arrays, masked arrays, memory maps and the FFT. Chapter 8 lists these topics as not measured.
- **Two honest options are named and not measured.** A signal-library filter and Numba are options for the loops that chapter 6 could not vectorize. The book makes no claim about their speed.
- **First drafts were wrong.** The author predicted several results before running the code, and the runs corrected them. The first k-means run in chapter 8 failed the author's own assertion, and that failure became a section. The guessed timings in chapter 8 were wrong by a factor of 5. Chapter 1 showed `owns=False` for a reshaped array, which the author did not expect. Chapters 5, 6 and 7 each had a number or a digit count that the run corrected.

## Layout

| Path | Content |
|---|---|
| `chapters/` | the eight chapters |
| `examples/` | one runnable module per chapter |
| `exercises/` | the exercises, and `solutions/` with the worked solutions |
| `scripts/check_snippets.py` | the checker |
| `scripts/build_pdf.py` | builds one PDF with page numbers, with headless Chrome |
| `artifacts/` | the built PDF |
