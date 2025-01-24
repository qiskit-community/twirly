#!/usr/bin/env python3
# This code is part of Twirly.
#
# This is proprietary IBM software for internal use only, do not distribute outside of IBM
# Unauthorized copying of this file is strictly prohibited.
#
# (C) Copyright IBM 2025.

"""Utility script to verify copyright file headers."""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

# regex for character encoding from PEP 263
pep263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
allow_path = re.compile(r"^[-_a-zA-Z0-9]+")

HEADER = """# This code is part of Twirly.
#
# This is proprietary IBM software for internal use only, do not distribute outside of IBM
# Unauthorized copying of this file is strictly prohibited.
#"""


def discover_files(
    roots: Iterable[str],
    extensions: set[str] = frozenset({".py", ".pyx", ".pxd"}),
    omit: str = "",
) -> Iterable[str]:
    """Find all .py, .pyx, .pxd files in a list of trees."""
    for code_path in roots:
        path = Path(code_path)
        if path.is_dir():
            # Recursively search for files with the specified extensions
            for file in path.rglob("*"):
                if file.suffix in extensions and not file.match(omit):
                    yield str(file)


def validate_header(file_path: str) -> tuple[str, bool, str]:
    """Validate the header for a single file."""
    with open(file_path, encoding="utf8") as fd:
        lines = fd.readlines()
    start = 0
    for index, line in enumerate(lines):
        if index > 5:
            return file_path, False, "Header not found in first 5 lines"
        if index <= 2 and pep263.match(line):
            return (
                file_path,
                False,
                "Unnecessary encoding specification (PEP 263, 3120)",
            )
        if line.strip().startswith(HEADER.split("\n", maxsplit=1)[0]):
            start = index
            break

    for idx, (actual, required) in enumerate(zip(lines[start:], HEADER.split("\n"))):
        if (actual := actual.strip()) != (required := required.strip()):
            return (
                file_path,
                False,
                f"Header line {1 + start + idx} '{actual}' does not match '{required}'.",
            )
    if not lines[start + 5].startswith("# (C) Copyright IBM 20"):
        return (file_path, False, "Header copyright line not found")
    return file_path, True, None


def main():
    default_path = Path(__file__).resolve().parent.parent / "project_name"

    parser = argparse.ArgumentParser(description="Check file headers.")
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        default=[default_path],
        help="Paths to scan; defaults to '../project_name' relative to the script location.",
    )
    parser.add_argument(
        "-o",
        "--omit",
        type=str,
        default="",
        help="Glob of files to omit.",
    )
    args = parser.parse_args()

    python_files = discover_files(map(str, args.paths), omit=args.omit)
    with ProcessPoolExecutor() as executor:
        results = executor.map(validate_header, python_files)

    failed_files = [(file_path, err) for file_path, success, err in results if not success]
    if failed_files:
        for file_path, error_message in failed_files:
            sys.stderr.write(f"{file_path} failed header check because:\n")
            sys.stderr.write(f"{error_message}\n\n")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
