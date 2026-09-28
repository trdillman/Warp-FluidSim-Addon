"""Whitewater kernels must write every ``out_*`` array they declare.

A declared-but-unwritten output (TASK-REVIEW-036: ``k_ww_compact_scatter``
took ``out_count`` while ``k_ww_final_count`` owned the count) falsely
signals which kernel owns a buffer.
"""

import ast
import pathlib

WHITEWATER = (pathlib.Path(__file__).resolve().parents[1]
              / "fluidSim_warp_blender51" / "sim" / "whitewater.py")


def _is_kernel(node):
    return any(
        ast.unparse(dec).split("(")[0] in ("wp.kernel", "kernel")
        for dec in node.decorator_list
    )


def _written_names(func):
    written = set()
    for node in ast.walk(func):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                    written.add(sub.value.id)
        # Atomics and helper calls that receive the array also count as writes.
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    written.add(arg.id)
    return written


def test_every_declared_whitewater_output_is_written():
    tree = ast.parse(WHITEWATER.read_text(encoding="utf-8"))
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef) and _is_kernel(n)]
    assert kernels, "no whitewater kernels found"

    unwritten = []
    for kernel in kernels:
        written = _written_names(kernel)
        unwritten.extend(
            f"{kernel.name}.{arg.arg}"
            for arg in kernel.args.args
            if arg.arg.startswith("out_") and arg.arg not in written
        )
    assert unwritten == []
