#!/usr/bin/env python3
"""
Robust Auto-Fixer for Line Lengths (E501).
Safely splits long strings, respecting f-string expressions.
"""

import re
import sys
from pathlib import Path

LINE_LIMIT = 88


def is_inside_fstring_expression(line: str, split_index: int) -> bool:
    """
    Determines if the split_index is inside an f-string expression {...}.
    This prevents splitting like: f"Value: {calc - " \n " 1}"
    """
    # Find the start of the string (simplified for common cases)
    match = re.search(r'f["\']', line)
    if not match:
        return False  # Not an f-string, safe to split (mostly)

    start_quote = match.start()

    # Count braces from the start of the string up to the split point
    # We only care about braces *inside* the string content
    balance = 0

    for i, char in enumerate(line):
        if i >= split_index:
            break

        if i < start_quote:
            continue

        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1

    # If balance > 0, we are inside a { ... } block
    return balance > 0


def fix_file(file_path: Path):
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    modified = False

    for i, line in enumerate(lines):
        if len(line) <= LINE_LIMIT:
            new_lines.append(line)
            continue

        # Check for string patterns we can split
        # Matches:  variable = "string content..." or return f"string..."
        # Group 1: Indentation + Variable/Code
        # Group 2: Quote type (", ', f", f')
        # Group 3: String content
        # Group 4: End quote
        match = re.match(r'^(\s*.*?)((?:f|r)?["\'])(.*)(["\']\s*[\),]?)$', line)

        if match:
            prefix = match.group(1)
            quote_start = match.group(2)  # e.g. f"
            content_str = match.group(3)
            suffix = match.group(4)  # e.g. ",

            # Calculate raw quote character ( " or ' )
            raw_quote = quote_start[-1]

            # Calculate available space
            current_len = (
                len(prefix) + len(quote_start) + len(content_str) + len(suffix)
            )
            excess = current_len - LINE_LIMIT

            if excess <= 0:
                new_lines.append(line)
                continue

            # Determine split point (try to split at space)
            # We want the first part to be around LINE_LIMIT length
            target_len = (
                LINE_LIMIT - len(prefix) - len(quote_start) - 2
            )  # -2 for closing quote and slash

            if target_len <= 10:  # Too cramped to split properly
                new_lines.append(line)
                continue

            # Find safe split index
            # Look for space closest to target_len from the left
            split_candidate = -1
            for j in range(target_len, 0, -1):
                if j < len(content_str) and content_str[j] == " ":
                    split_candidate = j
                    break

            if split_candidate == -1:
                # No spaces? Force split at limit
                split_candidate = target_len

            # --- SAFETY CHECK: Don't split inside f-string expressions ---
            if "f" in quote_start and is_inside_fstring_expression(
                line, len(prefix) + len(quote_start) + split_candidate
            ):
                print(f"⚠️  Skipping unsafe f-string split in {file_path.name}:{i + 1}")
                new_lines.append(line)
                continue
            # -------------------------------------------------------------

            part1 = content_str[:split_candidate]
            part2 = content_str[
                split_candidate:
            ]  # Include the space in second part or keep in first?
            # Usually better to keep space at end of line or start of next.

            # Construct new lines
            # Line 1: prefix + quote_start + part1 + " " (implicit concatenation)
            # OR use explicit backslash if inside function call?
            # Safer to use implicit string concatenation with parens,
            # but here we use standard split.

            # Approach: Close quote, newline, indent, open quote
            indent = " " * (len(prefix) + 4)  # Simple indent

            # Handle parenthesis wrapping if detected
            if prefix.strip().endswith("("):
                indent = " " * len(prefix)

            line1 = f"{prefix}{quote_start}{part1}{raw_quote}"
            line2 = f"{indent}{quote_start}{part2}{suffix}"

            new_lines.append(line1)
            new_lines.append(line2)
            modified = True
            print(f"Fixed line {i + 1} in {file_path.name}")
        else:
            new_lines.append(line)

    if modified:
        file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) < 2:
        print("Usage: python autofix_long_lines.py <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        fix_file(target)
    elif target.is_dir():
        for py_file in target.rglob("*.py"):
            fix_file(py_file)


if __name__ == "__main__":
    main()
