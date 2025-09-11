#!/usr/bin/env bash
# release.sh — bump version, clean, build, check, upload
# Usage:
#   ./release.sh --patch|--minor|--major|<NEW_VERSION> [--test]
# Examples:
#   ./release.sh --patch
#   ./release.sh 0.9.4 --test

set -euo pipefail

REPO="pypi" # default; use --test to switch to TestPyPI
BUMP_MODE=""
NEW_VERSION_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) REPO="testpypi"; shift ;;
    --patch|--minor|--major) BUMP_MODE="${1#--}"; shift ;;
    *)
      if [[ -z "${NEW_VERSION_ARG}" ]]; then
        NEW_VERSION_ARG="$1"; shift
      else
        echo "Unexpected argument: $1" >&2; exit 1
      fi
      ;;
  esac
done

if [[ ! -f "setup.py" ]]; then
  echo "setup.py not found. Run from project root." >&2
  exit 1
fi

# Determine and set the new version inside setup.py
python - <<'PY'
import re, sys
from pathlib import Path

path = Path("setup.py")
text = path.read_text(encoding="utf-8")

m = re.search(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', text)
if not m:
    print("Could not find version=... in setup.py", file=sys.stderr)
    sys.exit(1)

current = m.group(1)

import os
bump_mode = os.environ.get("BUMP_MODE","")
new_version_arg = os.environ.get("NEW_VERSION_ARG","").strip()

def bump(ver, part):
    nums = ver.split(".")
    # pad to 3
    while len(nums) < 3:
        nums.append("0")
    major, minor, patch = map(int, nums[:3])
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor += 1; patch = 0
    elif part == "major":
        major += 1; minor = 0; patch = 0
    else:
        raise ValueError("unknown bump part")
    return f"{major}.{minor}.{patch}"

if new_version_arg:
    new = new_version_arg
elif bump_mode:
    new = bump(current, bump_mode)
else:
    # default: patch bump
    new = bump(current, "patch")

new_text = re.sub(
    r'(version\s*=\s*[\'"])([^\'"]+)([\'"])',
    rf'\g<1>{new}\3',
    text,
    count=1
)
if new_text == text:
    print("Version not changed; aborting.", file=sys.stderr)
    sys.exit(1)

path.write_text(new_text, encoding="utf-8")
print(f"{current} -> {new}")
PY
export BUMP_MODE="${BUMP_MODE:-}"
export NEW_VERSION_ARG="${NEW_VERSION_ARG:-}"

# Clean previous builds
rm -rf build dist *.egg-info

# Ensure build tooling and documentation utilities
python -m pip install -U pip setuptools wheel build twine pdoc mkdocs mkdocs-material

# Regenerate API documentation
pdoc -d google -o docs/api_reference causal_pipe
test -s docs/api_reference/index.html

# Build the website → outputs to site/
mkdocs build --strict
mkdocs gh-deploy

# Build sdist + wheel
python -m build

# Check metadata
python -m twine check dist/*

# Upload
if [[ "${REPO}" == "testpypi" ]]; then
  python -m twine upload --repository testpypi dist/*
else
  python -m twine upload dist/*
fi

echo "Done."
