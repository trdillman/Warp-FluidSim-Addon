"""Kernel identity registry (dispatch contract: ``kernel_lookup``).

Every canonical kernel name from the synthesized manifest is registered here
and resolved to its Warp implementation. Size-suffixed OpenCL variants
(e.g. Jacobi8/16/32, ReduceSum128..1024) map to one parameterized Warp
implementation registered under all aliases; the suffix was an OpenCL
workgroup tuning knob that Warp's runtime replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    fn: Callable
    strategy: str = "flat"  # "flat" | "tiled" | "host_orchestrated"
    notes: str = ""


KERNEL_REGISTRY: dict[str, RegistryEntry] = {}

# original source namespace alias: paradigm* -> fluidSim*
SOURCE_ALIASES: dict[str, str] = {}


def register(*names: str, strategy: str = "flat", notes: str = ""):
    """Register a callable under one or more canonical kernel identities."""

    def deco(fn):
        requested = set()
        for n in names:
            if n in requested:
                raise ValueError(
                    f"FluidSim kernel identity {n!r} was requested more than once"
                )
            requested.add(n)
            if n in KERNEL_REGISTRY:
                previous = KERNEL_REGISTRY[n]
                raise ValueError(
                    f"FluidSim kernel identity {n!r} is already registered "
                    f"by {previous.fn!r}"
                )

        for n in names:
            KERNEL_REGISTRY[n] = RegistryEntry(n, fn, strategy, notes)
            if n.startswith("fluidSim"):
                SOURCE_ALIASES["paradigm" + n[len("fluidSim"):]] = n
        return fn

    return deco


def register_fn(fn: Callable, *names: str, strategy: str = "flat", notes: str = ""):
    register(*names, strategy=strategy, notes=notes)(fn)


def lookup(name: str) -> RegistryEntry:
    if name in KERNEL_REGISTRY:
        return KERNEL_REGISTRY[name]
    if name in SOURCE_ALIASES:
        return KERNEL_REGISTRY[SOURCE_ALIASES[name]]
    raise KeyError(f"Unknown FluidSim kernel identity: {name}")


def registered_names() -> list[str]:
    return sorted(KERNEL_REGISTRY)
