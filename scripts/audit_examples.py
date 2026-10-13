"""List code samples that break the example rules, for a person to read.

    uv run python scripts/audit_examples.py [files...]

Rules, from the review of chapter 1:
  chain     the sample builds data with `arange(...).reshape(...)`: it makes a view of an unnamed block, and the
            values equal their positions, so a reader cannot tell a value from a place
  dtype     the sample passes `dtype=` with no reason in the text next to it: a reader who types the obvious
            call gets other numbers
  early     the sample calls an np function before the `####` block that introduces it
  unsigned  the sample calls an np function that has no `####` block in the chapter (information only)

It never fails. It is an audit.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```python\n(.*?)```", re.S)
CALL = re.compile(r"\bnp\.((?:[A-Za-z_]\w*)(?:\.[A-Za-z_]\w*)*)\(")
HEAD = re.compile(r"^#### `np\.([\w.]+)`", re.M)


def files() -> list[pathlib.Path]:
    named = [pathlib.Path(a).resolve() for a in sys.argv[1:]]
    if named:
        return named
    return sorted([*(ROOT / "chapters").glob("*.md"), *(ROOT / "exercises").glob("*.md"),
                   *(ROOT / "exercises" / "solutions").glob("*.md")])


def main() -> None:
    hits = {"chain": 0, "dtype": 0, "early": 0, "unsigned": 0}
    for p in files():
        text = p.read_text(encoding="utf-8")
        intro = {m.group(1): m.start() for m in HEAD.finditer(text)}
        out = []
        for m in FENCE.finditer(text):
            code, line = m.group(1), text[:m.start()].count("\n") + 1
            if re.search(r"arange\([^)]*\)\s*\.reshape\(", code):
                out.append((line, "chain", "arange(...).reshape(...)")); hits["chain"] += 1
            for d in sorted(set(re.findall(r"dtype=np\.\w+", code))):
                out.append((line, "dtype", d)); hits["dtype"] += 1
            for name in sorted(set(CALL.findall(code))):
                key = name if name in intro else next((k for k in intro if name.startswith(k + ".")), None)
                if key is None:
                    if p.parent.name == "chapters":
                        out.append((line, "unsigned", f"np.{name}")); hits["unsigned"] += 1
                elif m.start() < intro[key]:
                    out.append((line, "early", f"np.{name} (introduced at line {text[:intro[key]].count(chr(10)) + 1})"))
                    hits["early"] += 1
        if out:
            print(p.relative_to(ROOT))
            for line, kind, what in out:
                print(f"  {line:5d}  {kind:9s} {what}")
    print("\n" + ", ".join(f"{k} {v}" for k, v in hits.items()))


if __name__ == "__main__":
    main()
