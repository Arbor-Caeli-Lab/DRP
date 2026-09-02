"""Static checks for the split, relocatable reproduction repository."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CHAPTERS = [f"Code_chapter{i:02d}" for i in range(1, 7)]
SCRIPT_SUFFIXES = {".py", ".R", ".r", ".Rmd", ".rmd", ".qmd"}

_DRIVE_PREFIX = r"[a-z]" + r":[\\/]"
ABSOLUTE_PATH = re.compile(r"(?i)(?<![a-z0-9])(?:" + _DRIVE_PREFIX + r"|/home/|/users/)")
LEGACY_PATH = re.compile(
    r"(?i)(?:\./(?:data|results)(?:[\\/]|$)|"
    r"Code_chapter\d{2}[\\/](?:data|results)(?:[\\/]|$))"
)


def check_layout(errors: list[str]) -> None:
    for chapter in CHAPTERS:
        for root in (REPO_ROOT / "data" / chapter, REPO_ROOT / "results" / chapter,
                     SCRIPT_ROOT / chapter):
            if not root.exists():
                errors.append(f"missing expected directory: {root.relative_to(REPO_ROOT)}")


def check_scripts(errors: list[str]) -> tuple[int, int]:
    checked = 0
    python_checked = 0
    for path in sorted(SCRIPT_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in SCRIPT_SUFFIXES:
            continue
        checked += 1
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)
        if path.resolve() != Path(__file__).resolve():
            for line_number, line in enumerate(text.splitlines(), start=1):
                if ABSOLUTE_PATH.search(line):
                    errors.append(f"absolute path in {relative}:{line_number}")
                if LEGACY_PATH.search(line):
                    errors.append(f"legacy relative path in {relative}:{line_number}")

        if path.suffix == ".py":
            python_checked += 1
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {relative}:{exc.lineno}: {exc.msg}")
    return checked, python_checked


def main() -> int:
    errors: list[str] = []
    check_layout(errors)
    checked, python_checked = check_scripts(errors)

    if errors:
        print("Path validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Path validation passed: {checked} scripts checked; {python_checked} Python files parsed.")
    print("All script paths are repository-relative/relocatable; no legacy path references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
