"""Chapter 8: k-means from the parts of chapters 4 and 5, checked against a plain loop."""
import timeit

import numpy as np

from examples.ch05_broadcasting import dist_dot_centered


def make_blobs(n_per: int, centers: np.ndarray, spread: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    X = np.concatenate([c + spread * g.standard_normal((n_per, centers.shape[1])) for c in centers])
    truth = np.repeat(np.arange(len(centers)), n_per)
    return X, truth


def kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    """Lloyd's algorithm. Returns (centroids, labels, iterations)."""
    g = np.random.default_rng(seed)
    C = X[g.choice(len(X), k, replace=False)].copy()
    labels = np.full(len(X), -1)
    for it in range(1, max_iter + 1):
        new_labels = dist_dot_centered(X, C).argmin(axis=1)      # chapter 5
        if np.array_equal(new_labels, labels):
            return C, labels, it - 1
        labels = new_labels
        sums = np.zeros_like(C)
        np.add.at(sums, labels, X)                               # chapter 4
        counts = np.bincount(labels, minlength=k)
        keep = counts > 0                                        # an empty cluster keeps its centroid
        C[keep] = sums[keep] / counts[keep, None]
    return C, labels, max_iter


def kmeans_loop(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, int]:
    """The same algorithm with explicit loops. Used only to check `kmeans`."""
    g = np.random.default_rng(seed)
    C = X[g.choice(len(X), k, replace=False)].copy()
    labels = [-1] * len(X)
    for it in range(1, max_iter + 1):
        new_labels = []
        for x in X:
            d = [float(((x - c) ** 2).sum()) for c in C]
            new_labels.append(d.index(min(d)))
        if new_labels == labels:
            return C, np.array(labels), it - 1
        labels = new_labels
        for j in range(k):
            members = [x for x, l in zip(X, labels) if l == j]
            if members:
                C[j] = np.mean(members, axis=0)
    return C, np.array(labels), max_iter


def inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances from each point to its centroid."""
    return float(((X - C[labels]) ** 2).sum())


def kmeans_best(X: np.ndarray, k: int, n_init: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Run `kmeans` from `n_init` starts and keep the run with the lowest inertia."""
    seeds = np.random.SeedSequence(seed).spawn(n_init)
    runs = [kmeans(X, k, seed=s) for s in seeds]
    return min(runs, key=lambda r: inertia(X, r[0], r[1]))


def purity(labels: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Fraction of points whose cluster's majority true label equals their own."""
    total = 0
    for j in range(k):
        t = truth[labels == j]
        if len(t):
            total += np.bincount(t).max()
    return total / len(truth)


def main() -> None:
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X, truth = make_blobs(300, centers, 0.8, seed=1)

    C, labels, _ = kmeans(X, 3, seed=1)
    Cl, labels_l, _ = kmeans_loop(X, 3, seed=1)
    assert np.array_equal(labels, labels_l) and np.allclose(C, Cl)

    single = [purity(kmeans(X, 3, seed=s)[1], truth, 3) > 0.95 for s in range(30)]
    best = [purity(kmeans_best(X, 3, 10, seed=s)[1], truth, 3) > 0.95 for s in range(30)]
    print(f"single start finds the blobs in {sum(single)} of 30 seeds")
    print(f"best of 10 starts finds them in {sum(best)} of 30 seeds")
    assert sum(best) == 30 and sum(single) < 30

    Xs, _ = make_blobs(300, centers, 0.8, seed=2)
    t_loop = min(timeit.repeat(lambda: kmeans_loop(Xs, 3, seed=1), number=1, repeat=2))
    t_vec = min(timeit.repeat(lambda: kmeans(Xs, 3, seed=1), number=1, repeat=5))
    print(f"loop / vectorized on 900 points: {t_loop / t_vec:.0f}x")
    assert t_loop / t_vec > 20


if __name__ == "__main__":
    main()
