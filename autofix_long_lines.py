import json
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path


def get_indentation(line: str) -> str:
    """Returns the whitespace at the start of the line."""
    return line[: len(line) - len(line.lstrip())]


def extract_limit_from_message(message: str) -> int:
    """
    Parses Ruff error message to find the configured line length limit.
    """
    match = re.search(r"Line too long \(\d+ > (\d+)\)", message)
    if match:
        return int(match.group(1))
    return 88  # Default fallback


def split_long_line(line_content: str, target_len: int) -> str:
    """
    Splits a line containing a long string or comment into two lines.
    """
    if len(line_content) <= target_len:
        return line_content

    indent = get_indentation(line_content)
    stripped = line_content.lstrip()

    # --- STRATEGY 1: Full-Line Comments ---
    if stripped.startswith("#"):
        # Skip special comments (shebang, encoding, pylint/ruff ignores)
        if stripped.startswith("#!") or "coding=" in stripped or "noqa" in stripped:
            return line_content

        split_idx = line_content.rfind(" ", 0, target_len)

        if split_idx < len(indent) + 2:
            return line_content

        prefix = line_content[:split_idx]
        suffix = line_content[split_idx + 1 :]

        return f"{prefix}\n{indent}# {suffix}"

    # --- STRATEGY 2: Strings (Implicit Concatenation) ---
    if " " not in line_content or ("'" not in line_content and '"' not in line_content):
        return line_content

    quote_char = '"' if '"' in line_content else "'"
    limit = target_len - 5
    if limit < 10:
        limit = 80

    split_idx = line_content.rfind(" ", 0, limit)

    if split_idx < len(indent) + 5:
        return line_content

    prefix = line_content[:split_idx]
    suffix = line_content[split_idx + 1 :]

    return f"{prefix} {quote_char}\n{indent}    {quote_char}{suffix}"


def fix_python_file(file_path: Path, errors: list):
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        modified = False
        errors.sort(key=lambda x: x["location"]["row"], reverse=True)

        for err in errors:
            if err["code"] != "E501":
                continue

            line_idx = err["location"]["row"] - 1
            limit = extract_limit_from_message(err["message"])

            if 0 <= line_idx < len(lines):
                original = lines[line_idx].rstrip("\n")
                fixed = split_long_line(original, target_len=limit)

                if fixed != original:
                    lines[line_idx] = fixed + "\n"
                    modified = True
                    print(f"  ✅ Fixed line {line_idx + 1} (Limit: {limit})")
                else:
                    print(f"  ⚠️  Skipped line {line_idx + 1} (Too complex)")

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")


def main():
    print("🚀 Running Ruff (using pyproject.toml rules)...")

    # SECURITY FIX: Resolve the full path to the executable
    # This prevents B607 (starting process with partial path)
    ruff_executable = shutil.which("ruff")
    if not ruff_executable:
        print("❌ Error: 'ruff' executable not found in PATH.")
        return

    try:
        # SECURITY FIX: Added nosec B603
        # We are passing a known executable and hardcoded arguments.
        result = subprocess.run(
            [ruff_executable, "check", ".", "--output-format", "json"],
            capture_output=True,
            text=True,
            check=False,
        )  # nosec B603
    except Exception as e:
        print(f"❌ Error executing ruff: {e}")
        return

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        if not result.stdout.strip():
            print("✅ No errors output by Ruff.")
        else:
            print("⚠️  Ruff output was not valid JSON.")
        return

    e501_errors = [e for e in data if e["code"] == "E501"]

    if not e501_errors:
        print("✅ No E501 (Line too long) errors found.")
        return

    files_map = {}
    for item in e501_errors:
        fname = item["filename"]
        if fname not in files_map:
            files_map[fname] = []
        files_map[fname].append(item)

    print(
        f"⚠️  Found {len(e501_errors)} line-length errors in {len(files_map)} files.\n"
    )

    for fname, errs in files_map.items():
        path = Path(fname)
        if not path.exists():
            continue

        print(f"📄 Checking {fname}...")
        if fname.endswith(".py"):
            fix_python_file(path, errs)
        elif fname.endswith(".ipynb"):
            print("  ⚠️  Skipping Notebook (Manual edit recommended).")
        else:
            print("  ℹ️  Skipping non-Python file.")

    print("\n✨ Done.")


if __name__ == "__main__":
    main()
