"""Deprecated alias for `mlenergy.data`.

`mlenergy_data` was the original import path for the ML.ENERGY data toolkit.
The canonical path is now `mlenergy.data`. This module aliases itself to the
real package so existing imports keep working, and emits a
DeprecationWarning on first import.
"""

import importlib
import pkgutil
import sys
import warnings

import mlenergy.data as _real

for _info in pkgutil.walk_packages(_real.__path__, prefix=f"{_real.__name__}."):
    _legacy = "mlenergy_data" + _info.name[len(_real.__name__) :]
    sys.modules[_legacy] = importlib.import_module(_info.name)

warnings.warn(
    "Import from `mlenergy.data` instead of `mlenergy_data`. "
    "The `mlenergy_data` alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _real
