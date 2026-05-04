"""Import / compile checks for every Python file in the release.

The tests do not execute the scripts (some require a GPU, a dataset, or both).
They only confirm each ``.py`` parses and that the spectranet package itself
imports cleanly.
"""

from __future__ import annotations
import os
import py_compile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _all_python_files():
    files = []
    for root in ["spectranet", "scripts", "baselines", "timing", "tests"]:
        d = REPO / root
        if d.exists():
            for p in d.rglob("*.py"):
                if "__pycache__" in p.parts:
                    continue
                files.append(p)
    return files


@pytest.mark.parametrize("path", _all_python_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)


def test_spectranet_imports():
    """Top-level `spectranet` API must load without optional GPU dependencies."""
    import spectranet  # noqa: F401
    from spectranet import SUNet2d, SpectralConfig, LpLoss  # noqa: F401
    from spectranet.data import load_dataset, MatReader  # noqa: F401
    from spectranet.utils import count_params, seed_all  # noqa: F401


def test_no_personal_identifiers_in_release():
    """One-strike anonymity guard: scan every text file in the release for
    deanonymizing tokens.  The regex is assembled at runtime from short
    fragments so this test file itself does not match its own pattern.
    """
    import re
    fragments = [
        "ehern" + "an8",
        "jup" + "yterhub",
        "enriquehern" + "andez",
        "GUL" + "F_SCEI",
    ]
    bad = re.compile("|".join(fragments), re.I)
    skip_dirs = {"checkpoints", "data", "figures", "__pycache__", ".pytest_cache"}
    skip_files = {"ANONYMITY.md", "test_imports.py"}    # legitimately mention the patterns
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if fname in skip_files:
                continue
            if not (fname.endswith(".py") or fname.endswith(".md") or fname.endswith(".sh")
                    or fname.endswith(".yaml") or fname.endswith(".csv") or fname.endswith(".toml")
                    or fname.endswith(".cff") or fname == ".gitignore"):
                continue
            path = Path(root) / fname
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            hits = list(bad.finditer(text))
            assert not hits, (
                f"personal-identifier hit in {path}: "
                f"{[h.group(0) for h in hits[:3]]}"
            )
