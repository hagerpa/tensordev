"""Private planning and execution support for wordwise computations.

Names are resolved lazily so importing the cheap device dispatcher on a CPU
does not load plan builders or the Pallas kernel.
"""

from __future__ import annotations

from importlib import import_module


_LAYOUT_NAMES = frozenset(
    {
        "DEFAULT_PLAN_LIMITS",
        "PlanEstimate",
        "PlanLimits",
        "PlanResourceError",
        "WordwiseBlockPlan",
        "WordwiseLayoutPlan",
        "build_layout_plan",
        "check_plan_resources",
        "clear_layout_plan_cache",
        "estimate_layout_plan",
    }
)
_PLAN_NAMES = frozenset({"PrefixGraphPlan", "PrefixGraphView"})

__all__ = sorted(_LAYOUT_NAMES | _PLAN_NAMES)


def __getattr__(name: str):
    if name in _LAYOUT_NAMES:
        module_name = "tensordev._wordwise.layout"
    elif name in _PLAN_NAMES:
        module_name = "tensordev._wordwise.plans"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
