"""Solver-version policy shared by bake execution and cache controls."""


def resumable_solver_version(cached: str, current: str) -> bool:
    """A checkpoint can resume only within the current solver major/minor."""
    try:
        cached_parts = tuple(int(value) for value in cached.split("."))
        current_parts = tuple(int(value) for value in current.split("."))
    except ValueError:
        return False
    return cached_parts[:2] == current_parts[:2]
