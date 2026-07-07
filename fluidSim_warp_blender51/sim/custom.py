"""Custom user Warp snippet support.

Port of the fluidSim custom OpenCL kernel feature to controlled Warp/Python
extension points (see custom_kernel_contract). Users provide the *body* of a
Warp kernel in Python syntax; the surrounding kernel is generated with the
documented context variables in scope. Multiple snippets per hook order are
supported (the original tool stopped after the first snippet; this port
executes all of them in declaration order, resolving the audit item).

Run-over contexts (variables available to the snippet body):

  VoxelTile / VoxelLinear:
    flat_voxel_index (int), leaf_index (int), local_x/local_y/local_z (int),
    global_x/global_y/global_z (int), world_position (wp.vec3),
    voxel_size (float), plus time context and declared arguments.
    Collision voxels are skipped before the body runs.

  Particles:
    particle_index (int), position (wp.vec3), velocity (wp.vec3),
    plus time context and declared arguments. Dead particles are skipped.

Time context (always available):
    frame (int), time (float), real_time_step (float),
    solver_time_step (float), time_scale (float), substeps (int)

Declared arguments:
  mode 'field'    -> the named solver field is passed as `wp.array` visible
                     under the argument's name. Known fields are listed in
                     FIELD_ACCESSORS.
  mode 'constant' -> a float/int uniform visible under `constant_name`.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field as dc_field

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from . import snippet_library
from .enums import (
    CustomKernelOrder,
    CustomKernelRunOver,
    CustomKernelArgumentMode,
    CustomKernelConstantType,
)


# Field name -> (dtype string, accessor on solver)
FIELD_ACCESSORS = {
    "velocity_u": ("float", lambda s: s.vel.u),
    "velocity_v": ("float", lambda s: s.vel.v),
    "velocity_w": ("float", lambda s: s.vel.w),
    "weight_u": ("float", lambda s: s.vel.wu),
    "weight_v": ("float", lambda s: s.vel.wv),
    "weight_w": ("float", lambda s: s.vel.ww),
    "temperature": ("float", lambda s: s.temperature.data),
    "viscosity": ("float", lambda s: s.viscosity_field.data),
    "density": ("float", lambda s: s.density.data),
    "pressure": ("float", lambda s: s.pressure_solver.pressure),
    "divergence": ("float", lambda s: s.pressure_solver.divergence),
    "surface_sdf": ("float", lambda s: s.surface_sdf.data),
    "collision_sdf": ("float", lambda s: s.collision_sdf.data),
    "churn": ("float", lambda s: s.churn.data),
    "voxel_type": ("int", lambda s: s.voxel_type.data),
    "voxel_counts": ("int", lambda s: s.voxel_counts.data),
    # Particle arrays (Particles run-over).
    "particle_position": ("vec3", lambda s: s.particles.position),
    "particle_velocity": ("vec3", lambda s: s.particles.velocity),
    "particle_age": ("float", lambda s: s.particles.attr("age")),
    "particle_life": ("float", lambda s: s.particles.attr("life")),
    "particle_temperature": ("float", lambda s: s.particles.attr("temperature")),
    "particle_viscosity": ("float", lambda s: s.particles.attr("viscosity")),
    "particle_density": ("float", lambda s: s.particles.attr("density")),
}

_DTYPE_DECL = {"float": "wp.array(dtype=float)",
               "int": "wp.array(dtype=wp.int32)",
               "vec3": "wp.array(dtype=wp.vec3)"}


@dataclass
class CustomArgument:
    mode: CustomKernelArgumentMode = CustomKernelArgumentMode.FIELD
    field: str = "temperature"
    constant_type: CustomKernelConstantType = CustomKernelConstantType.FLOAT
    constant_name: str = "value"
    constant_value: float = 0.0


@dataclass
class CustomSnippet:
    enabled: bool = True
    kernel_name: str = "user_kernel"
    order: CustomKernelOrder = CustomKernelOrder.FINAL_STEP
    run_over: CustomKernelRunOver = CustomKernelRunOver.VOXEL_LINEAR
    code: str = ""
    arguments: list = dc_field(default_factory=list)  # list[CustomArgument]
    # Layering window: restrict when this snippet fires so multiple snippets
    # can be combined at different times (0 = open ended, step = every Nth).
    frame_start: int = 0
    frame_end: int = 0
    frame_step: int = 1

    def active_on_frame(self, frame: int) -> bool:
        if self.frame_start and frame < self.frame_start:
            return False
        if self.frame_end and frame > self.frame_end:
            return False
        step = max(1, int(self.frame_step))
        if step > 1:
            base = self.frame_start if self.frame_start else 0
            if (frame - base) % step != 0:
                return False
        return True

    def argument_names(self):
        names = []
        for arg in self.arguments:
            if arg.mode == CustomKernelArgumentMode.FIELD:
                names.append(arg.field)
            else:
                names.append(arg.constant_name)
        return names

    def signature_params(self):
        """Returns (param_decl_list, launch_binder) for declared arguments."""
        decls = []
        binders = []
        for arg in self.arguments:
            if arg.mode == CustomKernelArgumentMode.FIELD:
                if arg.field not in FIELD_ACCESSORS:
                    raise ValueError(f"Unknown custom-kernel field '{arg.field}'")
                dtype, accessor = FIELD_ACCESSORS[arg.field]
                decls.append(f"{arg.field}: {_DTYPE_DECL[dtype]}")
                binders.append(("field", accessor))
            else:
                pyt = "float" if arg.constant_type == CustomKernelConstantType.FLOAT else "int"
                decls.append(f"{arg.constant_name}: {pyt}")
                value = arg.constant_value
                binders.append(("constant",
                                (float(value) if pyt == "float" else int(round(value)))))
        return decls, binders


_VOXEL_TEMPLATE = """
import warp as wp
wp.set_module_options({{"enable_backward": False}})
from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.fields import voxel_coord_of_flat, voxel_center_world

@wp.kernel
def {name}(
    _leaf_coords: wp.array(dtype=wp.vec3i),
    _num_leaves: wp.array(dtype=wp.int32),
    _voxel_type: wp.array(dtype=wp.int32),
    _origin: wp.vec3,
    voxel_size: float,
    frame: int,
    time: float,
    real_time_step: float,
    solver_time_step: float,
    time_scale: float,
    substeps: int,
    {extra_params}
):
    _tid = wp.tid()
    if _tid >= _num_leaves[0] * C.LEAF_VOLUME:
        return
    if (_voxel_type[_tid] & C.VOXEL_COLLISION) != 0:
        return
    flat_voxel_index = _tid
    leaf_index = _tid // C.LEAF_VOLUME
    _ijk = voxel_coord_of_flat(_leaf_coords, _tid)
    global_x = _ijk[0]
    global_y = _ijk[1]
    global_z = _ijk[2]
    local_x = global_x & C.LEAF_MASK
    local_y = global_y & C.LEAF_MASK
    local_z = global_z & C.LEAF_MASK
    world_position = voxel_center_world(_ijk, _origin, voxel_size)
{body}
"""

_PARTICLE_TEMPLATE = """
import warp as wp
wp.set_module_options({{"enable_backward": False}})
from fluidSim_warp_blender51.sim import constants as C

@wp.kernel
def {name}(
    _positions: wp.array(dtype=wp.vec3),
    _velocities: wp.array(dtype=wp.vec3),
    _alive: wp.array(dtype=wp.int32),
    _count: wp.array(dtype=wp.int32),
    voxel_size: float,
    frame: int,
    time: float,
    real_time_step: float,
    solver_time_step: float,
    time_scale: float,
    substeps: int,
    {extra_params}
):
    _tid = wp.tid()
    if _tid >= _count[0]:
        return
    if _alive[_tid] == 0:
        return
    particle_index = _tid
    position = _positions[_tid]
    velocity = _velocities[_tid]
{body}
"""


class CompiledSnippet:
    """A compiled user snippet ready to launch against a solver."""

    def __init__(self, snippet: CustomSnippet):
        self.snippet = snippet
        self.error = None
        self.kernel = None
        self._binders = []
        try:
            self._compile()
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.error = f"{type(exc).__name__}: {exc}"

    def _compile(self):
        sn = self.snippet
        decls, binders = sn.signature_params()
        self._binders = binders
        extra = ",\n    ".join(decls) if decls else "_unused_pad: float"
        if not decls:
            self._binders = [("constant", 0.0)]
        body = textwrap.indent(textwrap.dedent(sn.code).strip() or "pass", "    ")
        name = "custom_" + "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in sn.kernel_name
        )
        if sn.run_over == CustomKernelRunOver.PARTICLES:
            src = _PARTICLE_TEMPLATE.format(name=name, extra_params=extra, body=body)
        else:
            # VoxelTile and VoxelLinear share the flat launch; VoxelTile
            # additionally exposes leaf-local indices, which the template
            # always provides.
            src = _VOXEL_TEMPLATE.format(name=name, extra_params=extra, body=body)
        # Warp requires kernels to live in real source files (it re-reads the
        # file for codegen), so write the generated module to a snippet cache
        # directory and import it. Content-hashed names give deduplication
        # and stable Warp kernel caching across sessions.
        import hashlib
        import importlib.util
        import os
        import sys
        import tempfile

        digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
        mod_name = f"fluidsim_snippet_{name}_{digest}"
        snip_dir = os.path.join(tempfile.gettempdir(), "fluidsim_warp_snippets")
        os.makedirs(snip_dir, exist_ok=True)
        path = os.path.join(snip_dir, mod_name + ".py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
        self._module_name = mod_name
        if mod_name in sys.modules:
            module = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        self.kernel = module.__dict__[name]

    def launch(self, solver, time_ctx):
        if self.kernel is None:
            return
        sn = self.snippet
        extra_inputs = []
        for kind, binder in self._binders:
            if kind == "field":
                extra_inputs.append(binder(solver))
            else:
                extra_inputs.append(binder)
        common = [
            solver.grid.voxel_size,
            int(time_ctx["frame"]),
            float(time_ctx["time"]),
            float(time_ctx["real_time_step"]),
            float(time_ctx["solver_time_step"]),
            float(time_ctx["time_scale"]),
            int(time_ctx["substeps"]),
        ]
        if sn.run_over == CustomKernelRunOver.PARTICLES:
            particle_dim_fn = getattr(solver, "_np_dim", None)
            particle_dim = (particle_dim_fn() if callable(particle_dim_fn)
                            else solver.particles.capacity)
            wp.launch(
                self.kernel,
                dim=particle_dim,
                inputs=[
                    solver.particles.position,
                    solver.particles.velocity,
                    solver.particles.alive,
                    solver.particles.count,
                ] + common + extra_inputs,
                device=solver.device,
            )
        else:
            voxel_dim = max(1, int(getattr(solver, "_nvox", solver.grid.voxel_capacity)))
            wp.launch(
                self.kernel,
                dim=voxel_dim,
                inputs=[
                    solver.grid.leaf_coords,
                    solver.grid.num_leaves,
                    solver.voxel_type.data,
                    solver.grid.origin,
                ] + common + extra_inputs,
                device=solver.device,
            )


class SnippetManager:
    """Owns compiled snippets grouped by hook order."""

    def __init__(self):
        self._by_order = {order: [] for order in CustomKernelOrder}
        self.errors = []

    def set_snippets(self, snippets):
        self._by_order = {order: [] for order in CustomKernelOrder}
        self.errors = []
        for sn in snippets:
            if not sn.enabled:
                continue
            # Static validation gate: invalid snippets never reach codegen
            # or the running scene (Warp compile remains the final check).
            static_errors = [
                issue for issue in snippet_library.validate_snippet(
                    sn.code, sn.run_over.name, sn.argument_names())
                if issue.level == "ERROR"
            ]
            if static_errors:
                self.errors.append(
                    (sn.kernel_name, "; ".join(str(i) for i in static_errors)))
                continue
            compiled = CompiledSnippet(sn)
            if compiled.error:
                self.errors.append((sn.kernel_name, compiled.error))
            else:
                self._by_order[sn.order].append(compiled)

    def run(self, order, solver, time_ctx):
        frame = int(time_ctx.get("frame", 0))
        for compiled in self._by_order.get(order, ()):
            if not compiled.snippet.active_on_frame(frame):
                continue
            compiled.launch(solver, time_ctx)

    def has_any(self):
        return any(self._by_order.values())
