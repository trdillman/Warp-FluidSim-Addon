"""Cheap repeat kernel launches for the per-substep solver pipeline.

``wp.launch`` re-packs every argument on each call (~100 us for a ten-argument
kernel). The solver issues hundreds of launches per frame with the same
kernel on the same arrays, so the host spends more time packing arguments than
the GPU spends running them. :func:`launch` records a ``wp.Launch`` command
the first time a (kernel, device, array identity) combination is seen and
replays it afterwards, updating only the non-array arguments that changed.

The replayed kernel, dimensions, arguments and stream order are identical to a
plain ``wp.launch``, so results are bitwise unchanged. Anything unusual
(graph capture, CPU devices, launch options, a patched ``wp.launch``) falls
back to ``wp.launch`` itself.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import warp as wp

_ORIGINAL_LAUNCH = wp.launch
_MAX_ENTRIES = 8192
_SCALARS = (int, float, bool, np.integer, np.floating)
_cache: OrderedDict = OrderedDict()
_array_type = wp.array


def clear():
    """Drop every recorded launch (releases their module leases)."""
    _cache.clear()


def _plain(kernel, dim, inputs, outputs, device, kwargs):
    if outputs:
        return wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs,
                         device=device, **kwargs)
    return wp.launch(kernel, dim=dim, inputs=inputs, device=device, **kwargs)


def launch(kernel, dim, inputs=(), outputs=(), device=None, **kwargs):
    """Drop-in for ``wp.launch`` that replays recorded launches."""
    if (kwargs or wp.launch is not _ORIGINAL_LAUNCH
            or type(dim) is not int or dim <= 0 or device is None):
        return _plain(kernel, dim, inputs, outputs, device, kwargs)
    dev = wp.get_device(device)
    if not dev.is_cuda or dev.is_capturing:
        return _plain(kernel, dim, inputs, outputs, device, kwargs)

    args = [*inputs, *outputs] if outputs else inputs
    key = [kernel, dev.ordinal]
    scalar_slots = []
    other_slots = []
    for i, value in enumerate(args):
        if type(value) is _array_type:
            ptr = value.ptr
            if not ptr:
                return _plain(kernel, dim, inputs, outputs, device, kwargs)
            key.append((ptr, value.shape, value.strides))
        elif value is None:
            return _plain(kernel, dim, inputs, outputs, device, kwargs)
        else:
            key.append(None)
            (scalar_slots if isinstance(value, _SCALARS)
             else other_slots).append(i)
    key = tuple(key)

    entry = _cache.get(key)
    if entry is None:
        cmd = wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs,
                        device=device, record_cmd=True)
        # The command keeps the original argument objects for CPU APIC
        # capture only; dropping them stops the cache pinning device arrays
        # (the packed params hold raw pointers, never Python references).
        cmd.fwd_args = None
        if len(_cache) >= _MAX_ENTRIES:
            _cache.popitem(last=False)
        _cache[key] = [cmd, dim, [args[i] for i in scalar_slots]]
        cmd.launch()
        return None

    _cache.move_to_end(key)
    cmd, last_dim, last_scalars = entry
    if dim != last_dim:
        cmd.set_dim(dim)
        entry[1] = dim
    for slot, index in enumerate(scalar_slots):
        value = args[index]
        if value != last_scalars[slot]:
            cmd.set_param_at_index(index, value)
            last_scalars[slot] = value
    for index in other_slots:
        cmd.set_param_at_index(index, args[index])
    cmd.launch()
    return None
