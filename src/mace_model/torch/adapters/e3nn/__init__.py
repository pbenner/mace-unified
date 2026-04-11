from __future__ import annotations

from importlib import import_module

from .irreps import Irrep, Irreps

_OPTIMIZATION_DEFAULTS = {"jit_script_fx": False}


def get_optimization_defaults():
    return dict(_OPTIMIZATION_DEFAULTS)


def set_optimization_defaults(**kwargs):
    _OPTIMIZATION_DEFAULTS.update(kwargs)


__all__ = [
    "Irrep",
    "Irreps",
    "math",
    "nn",
    "o3",
    "util",
    "get_optimization_defaults",
    "set_optimization_defaults",
]


def __getattr__(name: str):
    if name in {"math", "nn", "o3", "util"}:
        return import_module(f".{name}", __name__)
    if name in {
        "Irrep",
        "Irreps",
        "get_optimization_defaults",
        "set_optimization_defaults",
    }:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
