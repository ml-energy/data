"""`mlenergy_data` is a deprecated alias of `mlenergy.data`.

These tests use `importlib` rather than `from mlenergy_data import ...` so that
static type checkers do not need to resolve the deprecated alias surface. The
alias is opaque to static tooling by design (see the `mlenergy_data/__init__.py`
docstring); these tests exist to catch runtime regressions.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def _reimport_mlenergy_data():
    sys.modules.pop("mlenergy_data", None)
    return importlib.import_module("mlenergy_data")


def test_alias_resolves_to_canonical():
    import mlenergy.data

    md = _reimport_mlenergy_data()
    assert md is mlenergy.data


def test_alias_emits_deprecation_warning():
    sys.modules.pop("mlenergy_data", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("mlenergy_data")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning on import of mlenergy_data"
    assert "mlenergy.data" in str(deprecations[0].message)


def test_top_level_symbols_via_alias():
    import mlenergy.data

    md = _reimport_mlenergy_data()
    for name in ("LLMRun", "LLMRuns", "DiffusionRun", "DiffusionRuns"):
        assert getattr(md, name) is getattr(mlenergy.data, name)


def test_dotted_submodule_paths_via_alias():
    from mlenergy.data.records.runs import LLMRun

    _reimport_mlenergy_data()
    legacy_runs = importlib.import_module("mlenergy_data.records.runs")
    assert legacy_runs.LLMRun is LLMRun


def test_version_is_shared():
    import mlenergy.data

    md = _reimport_mlenergy_data()
    assert md.__version__ == mlenergy.data.__version__
