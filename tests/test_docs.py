"""Test that Python code blocks in documentation files are runnable.

Uses pytest-examples to discover fenced Python code blocks in markdown files
and execute them.  All blocks within a single file share a chained namespace,
so a variable defined in an earlier block is visible to later blocks (matching
the sequential reading order of the guide).

Data is loaded from HF Hub by the code blocks themselves (cached locally
after first download).

Code fence annotations:
    test="skip"        Always skip (e.g. placeholder paths, heavy bulk downloads).
    test="skip-bulk"   Skip unless ``--run-bulk`` is passed.  These blocks
                       trigger bulk raw-file downloads from HF Hub.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-file namespace, keyed by resolved path.
# All code blocks within a file chain through this shared dict so that
# variables defined in earlier blocks are available to later ones.
_file_ns: dict[Path, dict] = {}


def _init_file_ns(path: Path) -> dict:
    """Build the initial namespace for a file's code blocks."""
    ns: dict = {}

    # The ITL example in the guide uses `itl_samples_s` without defining it.
    if "guide" in str(path):
        ns["itl_samples_s"] = np.random.default_rng(0).exponential(0.02, size=500)

    return ns


@pytest.mark.parametrize(
    "example",
    find_examples(
        REPO_ROOT / "docs",
        REPO_ROOT / "README.md",
        REPO_ROOT / "data_publishing" / "DATASET_CARD.md",
    ),
    ids=str,
)
def test_docs(example: CodeExample, eval_example: EvalExample, request) -> None:
    settings = example.prefix_settings()

    # Ensure file namespace is initialized before any skip, so later blocks
    # in the same file still get the injected globals.
    path = example.path.resolve()
    if path not in _file_ns:
        _file_ns[path] = _init_file_ns(path)

    # Honor skip annotations.
    test_mode = settings.get("test", "")
    if test_mode == "skip":
        pytest.skip('test="skip" in code fence')
    if test_mode == "skip-bulk" and not request.config.getoption("--run-bulk"):
        pytest.skip("needs --run-bulk")

    ns = eval_example.run(example, module_globals=dict(_file_ns[path]))
    _file_ns[path].update(ns)
