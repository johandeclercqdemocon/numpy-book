"""Print the signature of a NumPy callable exactly as the installed NumPy reports it.

Python functions have a real signature (`inspect.signature`). C functions and ufuncs do not, so for
those the first line of the docstring is the signature that NumPy documents.
"""
import inspect

import numpy as np


def signature_of(obj, name: str | None = None) -> str:
    """Return `name(args)` for `obj`, from inspect when possible and from the docstring otherwise."""
    name = name or getattr(obj, "__name__", repr(obj))
    try:
        return f"{name}{inspect.signature(obj)}"
    except (ValueError, TypeError):
        first = (obj.__doc__ or "").strip().splitlines()[0]
        return first.replace("a.", f"{name.rsplit('.', 1)[0]}.", 1) if first.startswith("a.") else first


def show(*items) -> None:
    """Print one signature per line. Each item is a callable or a (callable, name) pair."""
    for item in items:
        obj, name = item if isinstance(item, tuple) else (item, None)
        print(signature_of(obj, name))


def show_own(module, *names: str) -> None:
    """Print `name(args) -> result` and the first line of the docstring for each function of this book."""
    for name in names:
        f = getattr(module, name)
        doc = (f.__doc__ or "").strip().splitlines()
        print(f"{name}{inspect.signature(f)}")
        if doc:
            print(f"    {doc[0]}")


def resolve(path: str):
    """Turn a dotted name such as `np.add.at` or `np.random.Generator.integers` into the object."""
    root, *rest = path.split(".")
    if root != "np":
        raise ValueError(f"{path}: expected a name that starts with np.")
    obj = np
    for part in rest:
        obj = getattr(obj, part)
    return obj


def parameters(obj) -> list[tuple[str, str | None]]:
    """(name, default) for each parameter a caller can pass. The default is its text, or None if required.

    Python callables come from `inspect.signature`. C callables and ufuncs have no signature object, so
    their parameters are read from the first line of the docstring, which is the signature NumPy documents.
    """
    try:
        sig = inspect.signature(obj)
        return [(p.name, None if p.default is p.empty else repr(p.default))
                for p in sig.parameters.values()
                if p.kind is not p.VAR_KEYWORD and p.name != "self"]
    except (ValueError, TypeError):
        pass
    first = (obj.__doc__ or "").strip().splitlines()[0]
    inner = first[first.index("(") + 1:first.rindex(")")]
    out, depth, cur = [], 0, ""
    for ch in inner.replace("[", "").replace("]", ""):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    result = []
    for c in out:
        name, _, default = c.partition("=")
        name = name.strip().lstrip("*")
        if name and name not in ("/", "self", "kwargs"):
            result.append((name, default.strip() or None))
    return result


def parameter_names(obj) -> list[str]:
    """Names of the parameters that a caller can pass, in order."""
    return [name for name, _ in parameters(obj)]
