#!/usr/bin/env python3
"""Patch version strings across all packages from a git tag.

Usage:
    python scripts/set_release_version.py v0.1.0
    python scripts/set_release_version.py   # reads from git describe

This patches the version in:
  - openvaf_jax/openvaf_py/pyproject.toml + Cargo.toml
  - openvaf_jax/osdi_py/pyproject.toml + Cargo.toml
  - vajax/sparse/pyproject.toml

vajax itself uses hatch-vcs (automatic from git tags), so it doesn't need patching.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files to patch: (path, [(pattern, replacement_template)])
# replacement_template uses {version} placeholder
PATCH_TARGETS = [
    (
        REPO_ROOT / "openvaf_jax/openvaf_py/pyproject.toml",
        [(r'^version\s*=\s*"[^"]*"', 'version = "{version}"')],
    ),
    (
        REPO_ROOT / "openvaf_jax/openvaf_py/Cargo.toml",
        [(r'^version\s*=\s*"[^"]*"', 'version = "{version}"')],
    ),
    (
        REPO_ROOT / "openvaf_jax/osdi_py/pyproject.toml",
        [(r'^version\s*=\s*"[^"]*"', 'version = "{version}"')],
    ),
    (
        REPO_ROOT / "openvaf_jax/osdi_py/Cargo.toml",
        [(r'^version\s*=\s*"[^"]*"', 'version = "{version}"')],
    ),
    (
        REPO_ROOT / "vajax/sparse/pyproject.toml",
        [(r'^version\s*=\s*"[^"]*"', 'version = "{version}"')],
    ),
]


def get_version_from_tag(tag: str) -> str:
    """Extract PEP 440 version from a git tag like 'v0.1.0'."""
    version = tag.lstrip("v")
    # Validate it looks like a version
    if not re.match(r"^\d+\.\d+\.\d+", version):
        raise ValueError(f"Tag '{tag}' doesn't look like a version (expected vX.Y.Z)")
    return version


def get_version_from_git() -> str:
    """Get version from git describe."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--match", "v*"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git describe failed: {result.stderr.strip()}")
    return get_version_from_tag(result.stdout.strip().split("-")[0])


def patch_file(path: Path, patterns: list[tuple[str, str]], version: str) -> bool:
    """Patch version strings in a file. Returns True if changes were made."""
    if not path.exists():
        print(f"  SKIP {path.relative_to(REPO_ROOT)} (not found)")
        return False

    content = path.read_text()
    original = content
    for pattern, template in patterns:
        replacement = template.format(version=version)
        content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)

    if content != original:
        path.write_text(content)
        print(f"  PATCH {path.relative_to(REPO_ROOT)} -> {version}")
        return True
    else:
        print(f"  OK   {path.relative_to(REPO_ROOT)} (already {version})")
        return False


def main() -> int:
    if len(sys.argv) > 1:
        version = get_version_from_tag(sys.argv[1])
    else:
        version = get_version_from_git()

    print(f"Setting version: {version}")
    print()

    changed = 0
    for path, patterns in PATCH_TARGETS:
        if patch_file(path, patterns, version):
            changed += 1

    print()
    print(f"Done: {changed} files patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
