"""Chapter 4: indexing traps and a group-by sum three ways."""
import timeit

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


def best(f, repeat=3):
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main() -> None:
    rng = np.random.default_rng(0)
    n, k = 1_000_000, 100
    keys = rng.integers(0, k, n)
    values = rng.random(n)
    ref = group_sum_bincount(keys, values, k)
    for f in (group_sum_loop, group_sum_at):
        assert np.allclose(f(keys, values, k), ref)
    t_loop = best(lambda: group_sum_loop(keys, values, k))
    t_at = best(lambda: group_sum_at(keys, values, k))
    t_bc = best(lambda: group_sum_bincount(keys, values, k))
    print(f"loop / add.at   {t_loop / t_at:6.1f}x")
    print(f"add.at / bincount {t_at / t_bc:6.1f}x")
    assert t_loop / t_at > 5
    assert t_loop / t_bc > 5


if __name__ == "__main__":
    main()
