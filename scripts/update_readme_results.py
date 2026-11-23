"""Regenerate the README results table from the files in results/.

Usage:
    python scripts/update_readme_results.py          # rewrite README.md
    python scripts/update_readme_results.py --check  # exit 1 if README is out of date
"""

from __future__ import annotations

import argparse
import sys

from biomedqa.config import REPO_ROOT
from biomedqa.fileio import atomic_write_text
from biomedqa.reporting import render_table, replace_block


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    readme = REPO_ROOT / "README.md"
    old = readme.read_text(encoding="utf-8")
    new = replace_block(old, render_table())
    if args.check:
        if new != old:
            print("README results table is out of date: run scripts/update_readme_results.py")
            return 1
        print("README results table is up to date")
        return 0
    atomic_write_text(readme, new)
    print("README results table updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
