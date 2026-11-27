"""P0 target-runtime APIs, checked CUDA context and bounded native RTX feasibility.

Run in a private, automatically admitted Blender window for the complete probe.
This exercises API capabilities; the OptiX triangle is not an FSPT renderer or
an image-quality, throughput, zero-copy-display, or release qualification.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from threading import current_thread, main_thread

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as p1  # noqa: E402

ENGINE = "FLUIDSIM_FSPT_P0_PROBE"
REPORT = {}
OUTPUT = None
GRAPHICS_DONE = False
CALLS = {"render": 0, "view_update": 0, "view_draw": 0, "timer": 0}
wp = None  # Bound to the observed add-on's certified Warp import before compilation.
PASS_SPECS = (
    ("Combined", 4, "RGBA", "COLOR"),
    ("Alpha", 1, "A", "VALUE"),
    ("Depth", 1, "Z", "VALUE"),
    ("Normal", 3, "XYZ", "VECTOR"),
    ("Albedo", 3, "RGB", "COLOR"),
    ("Emission", 3, "RGB", "COLOR"),
    ("Environment", 3, "RGB", "COLOR"),
    ("Direct", 3, "RGB", "COLOR"),
    ("Indirect", 3, "RGB", "COLOR"),
    ("Transmission", 3, "RGB", "COLOR"),
)


def _capability_probe(out: wp.array(dtype=float)):
    out[0] = 3.0


def _signature(call):
    try:
        return str(inspect.signature(call))
    except (TypeError, ValueError):
        return None


def _symbol(call):
    return {
        "signature": _signature(call),
        "doc": (getattr(call, "__doc__", "") or "")[:4000],
    }


def _pattern(width, height):
    values = np.empty((height, width, 4), dtype=np.float32)
    values[..., 0] = np.linspace(0.02, 0.8, width, dtype=np.float32)[None, :]
    values[..., 1] = np.linspace(0.04, 0.6, height, dtype=np.float32)[:, None]
    values[..., 2] = 0.18
    values[..., 3] = 1.0
    return values


def _callback_thread(name, *, require_main=False):
    thread = current_thread()
    REPORT.setdefault("callback_threads", {})[name] = {
        "name": thread.name,
        "ident": thread.ident,
        "is_main": thread is main_thread(),
    }
    if require_main:
        assert thread is main_thread(), (name, thread.name)


def _read_image(path, *, encoded=False):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if encoded:
            image.colorspace_settings.name = "Non-Color"
        values = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        return values.reshape(image.size[1], image.size[0], 4)
    finally:
        bpy.data.images.remove(image)


def _graphics_probe(engine, context, scene):
    import gpu
    from gpu_extras.batch import batch_for_shader
    from gpu_extras.presets import draw_texture_2d
    from mathutils import Matrix

    assert current_thread() is main_thread()
    pixels = _pattern(8, 8)
    texture = gpu.types.GPUTexture(
        (8, 8),
        format="RGBA32F",
        data=gpu.types.Buffer("FLOAT", pixels.size, pixels.ravel()),
    )
    uploaded = np.asarray(texture.read(), dtype=np.float32).reshape(8, 8, 4)
    np.testing.assert_array_equal(uploaded, pixels)
    backend = {
        name: getattr(gpu.platform, name)()
        for name in (
            "backend_type_get",
            "device_type_get",
            "vendor_get",
            "renderer_get",
            "version_get",
        )
    }
    assert backend["backend_type_get"] in {"OPENGL", "VULKAN", "METAL"}, backend
    offscreen = gpu.types.GPUOffScreen(8, 8, format="RGBA32F")
    old_blend = gpu.state.blend_get()
    old_viewport = gpu.state.viewport_get()
    try:
        with offscreen.bind(), gpu.matrix.push_pop():
            gpu.state.viewport_set(0, 0, 8, 8)
            gpu.state.blend_set("NONE")
            gpu.state.active_framebuffer_get().clear(color=(0, 0, 0, 0))
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            shader = gpu.shader.from_builtin("IMAGE")
            batch = batch_for_shader(
                shader,
                "TRI_FAN",
                {
                    "pos": ((-1, -1), (1, -1), (1, 1), (-1, 1)),
                    "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1)),
                },
            )
            shader.bind()
            shader.uniform_sampler("image", texture)
            batch.draw(shader)
            drawn = np.asarray(
                offscreen.texture_color.read(), dtype=np.float32
            ).reshape(8, 8, 4)
            np.testing.assert_allclose(drawn, pixels, rtol=0, atol=1e-6)
    finally:
        offscreen.free()
        gpu.state.viewport_set(*old_viewport)
        gpu.state.blend_set(old_blend)
    # Execute the public managed-shader binding in its documented engine draw
    # context. Its presence is distinct from color-transform parity below.
    engine.bind_display_space_shader(scene)
    engine.unbind_display_space_shader()
    draw_texture_2d(texture, (0, 0), context.region.width, context.region.height)
    path = OUTPUT.with_name(OUTPUT.stem + "-rgba32f.npy")
    np.save(path, drawn, allow_pickle=False)
    texture_api = sorted(
        name for name in dir(type(texture)) if not name.startswith("__")
    )
    REPORT["graphics"] = {
        "status": "VERIFIED",
        "platform": backend,
        "context": "actual RenderEngine.view_draw on Blender main thread",
        "texture_format": texture.format,
        "shape": list(drawn.shape),
        "maximum_error": float(np.max(np.abs(drawn - pixels))),
        "texture_members": texture_api,
        "public_native_handle_candidates": {
            name: hasattr(texture, name)
            for name in (
                "bindcode",
                "as_pointer",
                "native_handle",
                "vulkan_image",
                "cuda_array",
            )
        },
        "raw_rgba": {"path": str(path), "sha256": p1.sha(path)},
        "api": {
            "GPUTexture": _symbol(gpu.types.GPUTexture),
            "GPUTexture.read": _symbol(gpu.types.GPUTexture.read),
            "GPUOffScreen": _symbol(gpu.types.GPUOffScreen),
            "GPUOffScreen.bind": _symbol(gpu.types.GPUOffScreen.bind),
            "Buffer": _symbol(gpu.types.Buffer),
            "GPUBatch.draw": _symbol(gpu.types.GPUBatch.draw),
        },
        "transport_scope": (
            "owned host RGBA upload plus Blender texture draw/readback; "
            "no CUDA image sharing"
        ),
        "display_shader_binding_executed": True,
        "display_transform_scope": (
            "EXR/PNG Standard-sRGB output is checked independently; "
            "viewport transform parity is not inferred"
        ),
    }


class ProbeEngine(bpy.types.RenderEngine):
    bl_idname = ENGINE
    bl_label = "FSPT P0 API probe"
    bl_use_preview = False

    def update_render_passes(self, scene, renderlayer):
        _callback_thread("update_render_passes")
        CALLS["update_render_passes"] = CALLS.get("update_render_passes", 0) + 1
        for name, channels, identifiers, kind in PASS_SPECS:
            self.register_pass(scene, renderlayer, name, channels, identifiers, kind)

    def render(self, depsgraph):
        _callback_thread("render")
        CALLS["render"] += 1
        scene = depsgraph.scene
        width, height = scene.render.resolution_x, scene.render.resolution_y
        for name, channels, identifiers, _kind in PASS_SPECS[1:]:
            self.add_pass(name, channels, identifiers, layer=depsgraph.view_layer.name)
        result = self.begin_result(0, 0, width, height, layer=depsgraph.view_layer.name)
        pattern = _pattern(width, height).reshape(-1, 4)
        values = {
            "Combined": pattern,
            "Alpha": pattern[:, 3:4],
            "Depth": np.full((width * height, 1), 2, dtype=np.float32),
            "Normal": np.tile(
                np.array((0, 0, 1), dtype=np.float32), (width * height, 1)
            ),
            "Albedo": pattern[:, :3],
            **{
                name: pattern[:, :3] * factor
                for name, factor in (
                    ("Emission", 0.1),
                    ("Environment", 0.2),
                    ("Direct", 0.3),
                    ("Indirect", 0.4),
                    ("Transmission", 0.15),
                )
            },
        }
        observed = []
        for name, channels, identifiers, _kind in PASS_SPECS:
            render_pass = result.layers[0].passes[name]
            assert render_pass.channels == channels
            render_pass.rect = values[name].tolist()
            actual = np.asarray(render_pass.rect, dtype=np.float32).reshape(
                -1, channels
            )
            np.testing.assert_allclose(actual, values[name], rtol=0, atol=1e-6)
            observed.append(
                {
                    "name": name,
                    "channels": channels,
                    "channel_ids": identifiers,
                    "maximum_write_read_error": float(
                        np.max(np.abs(actual - values[name]))
                    ),
                }
            )
        REPORT["pass_apis"] = {
            "status": "VERIFIED",
            "passes": observed,
            "scope": (
                "synthetic register_pass/add_pass/write/read transport; "
                "FSPT path semantics and multilayer file export NOT_RUN"
            ),
        }
        self.update_result(result)
        self.update_progress(1.0)
        self.end_result(result)

    def view_update(self, context, depsgraph):
        _callback_thread("view_update", require_main=True)
        CALLS["view_update"] += 1

    def view_draw(self, context, depsgraph):
        global GRAPHICS_DONE
        _callback_thread("view_draw", require_main=True)
        CALLS["view_draw"] += 1
        if GRAPHICS_DONE or REPORT.get("error"):
            return
        try:
            _graphics_probe(self, context, depsgraph.scene)
            self.update_stats("FSPT P0", "Verified RGBA32F API probe")
            self.tag_redraw()
            GRAPHICS_DONE = True
        except BaseException:
            REPORT.update(status="FAILED", error=traceback.format_exc())


def inspect_blender_apis():
    methods = (
        "begin_result",
        "update_result",
        "end_result",
        "update_progress",
        "tag_redraw",
        "tag_update",
        "bind_display_space_shader",
        "unbind_display_space_shader",
        "update_stats",
        "add_pass",
        "register_pass",
        "update_render_passes",
    )
    rna = bpy.types.RenderEngine.bl_rna.functions
    observed = {}
    for name in methods:
        function = rna.get(name)
        assert function is not None, f"Blender RenderEngine API missing: {name}"
        observed[name] = [
            {
                "name": p.identifier,
                "type": p.type,
                "output": p.is_output,
                "required": p.is_required,
            }
            for p in function.parameters
        ]
    view = bpy.context.preferences.view
    return {
        "RenderEngine": observed,
        "timers": {
            name: _symbol(getattr(bpy.app.timers, name))
            for name in ("register", "unregister", "is_registered")
        },
        "depsgraph_update": _symbol(bpy.types.Depsgraph.update),
        "evaluated_object": _symbol(bpy.types.ID.evaluated_get),
        "camera_calc_matrix": _symbol(bpy.types.Object.calc_matrix_camera),
        "view_layer_update_render_passes": _symbol(
            bpy.types.ViewLayer.update_render_passes
        ),
        "depth_of_field": sorted(bpy.types.CameraDOFSettings.bl_rna.properties.keys()),
        "motion_blur_render_properties": sorted(
            name
            for name in bpy.types.RenderSettings.bl_rna.properties.keys()
            if "motion_blur" in name
        ),
        "region_perspective_matrix": "perspective_matrix"
        in bpy.types.RegionView3D.bl_rna.properties,
        "show_splash": {
            "available": hasattr(view, "show_splash"),
            "value": getattr(view, "show_splash", None),
        },
        "event_simulate": {
            "available": hasattr(bpy.types.Window, "event_simulate"),
            "api": _symbol(getattr(bpy.types.Window, "event_simulate", None)),
            "launch_flag_present": "--enable-event-simulate" in sys.argv,
        },
        "source": "runtime RNA and built-in docstrings in observed Blender executable",
    }


def _authored_properties(owner, names):
    values = {}
    for name in names:
        if not hasattr(owner, name):
            values[name] = {"available": False}
            continue
        value = getattr(owner, name)
        if isinstance(value, bpy.types.ID):
            value = {"name": value.name, "type": type(value).__name__}
        elif not (value is None or isinstance(value, (str, bool, int, float))):
            value = str(value)
        values[name] = value
    return values


def inspect_authored_demos(paths):
    """Read appended authored scenes without making them active or solving them."""
    observations = []
    for path in paths:
        assert path.is_file(), f"Missing representative authored demo: {path}"
        with bpy.data.libraries.load(str(path), link=False) as (source, target):
            target.scenes = source.scenes[:]
        assert target.scenes
        try:
            scenes = []
            for scene in target.scenes:
                camera = scene.camera
                domain = scene.fluidsim.domain_object
                scenes.append(
                    {
                        "name": scene.name,
                        "authored_engine": scene.render.engine,
                        "frame_range": [scene.frame_start, scene.frame_end],
                        "domain": None
                        if domain is None
                        else {
                            "name": domain.name,
                            "type": domain.type,
                            "role": getattr(
                                getattr(domain, "fluidsim", None), "role", None
                            ),
                            "identity_source": "scene.fluidsim.domain_object",
                        },
                        "camera": None
                        if camera is None
                        else {
                            "name": camera.name,
                            "data": _authored_properties(
                                camera.data,
                                (
                                    "type",
                                    "lens",
                                    "sensor_width",
                                    "sensor_height",
                                    "sensor_fit",
                                    "ortho_scale",
                                    "shift_x",
                                    "shift_y",
                                    "clip_start",
                                    "clip_end",
                                ),
                            ),
                            "depth_of_field": _authored_properties(
                                camera.data.dof,
                                (
                                    "use_dof",
                                    "focus_distance",
                                    "focus_object",
                                    "aperture_fstop",
                                    "aperture_blades",
                                    "aperture_rotation",
                                    "aperture_ratio",
                                ),
                            ),
                            "animated": bool(
                                camera.animation_data and camera.animation_data.action
                            ),
                        },
                        "motion_blur": _authored_properties(
                            scene.render,
                            (
                                "use_motion_blur",
                                "motion_blur_shutter",
                                "motion_blur_position",
                            ),
                        ),
                        "cycles_motion_blur": _authored_properties(
                            scene.cycles,
                            (
                                "motion_blur_position",
                                "rolling_shutter_type",
                                "rolling_shutter_duration",
                            ),
                        )
                        if hasattr(scene, "cycles")
                        else None,
                    }
                )
            observations.append(
                {"path": str(path), "sha256": p1.sha(path), "scenes": scenes}
            )
        finally:
            for scene in target.scenes:
                if scene is not None:
                    bpy.data.scenes.remove(scene)
    return {
        "status": "VERIFIED",
        "demos": observations,
        "scope": (
            "actual authored scene properties; scenes appended but "
            "never activated or solved"
        ),
        "final_contract": {
            "depth_of_field": (
                "required Final endpoint; "
                "API/property inventory does not prove FSPT support"
            ),
            "motion_blur": (
                "required Final endpoint; "
                "API/property inventory does not prove FSPT support"
            ),
            "live": "DOF and motion blur may be disabled by the Live preset",
        },
    }


def probe_render_result(scene):
    assert scene.camera is not None
    matrix = scene.camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=16, y=16, scale_x=1, scale_y=1
    )
    REPORT["camera_matrix_execution"] = [list(row) for row in matrix]
    scene.render.engine = ENGINE
    bpy.context.view_layer.update_render_passes()
    assert CALLS.get("update_render_passes", 0) > 0
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.display_settings.display_device = "sRGB"
    scene.render.image_settings.color_mode = "RGBA"
    folder = OUTPUT.parent
    exr, png = folder / (OUTPUT.stem + "-api.exr"), folder / (OUTPUT.stem + "-api.png")
    scene.render.filepath = str(exr)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    assert bpy.ops.render.render(write_still=True) == {"FINISHED"}
    assert CALLS["render"] == 1 and exr.is_file()
    linear = _read_image(exr)
    expected = _pattern(16, 16)
    np.testing.assert_allclose(linear, expected, rtol=0, atol=1e-6)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    bpy.data.images["Render Result"].save_render(str(png), scene=scene)
    encoded = _read_image(png, encoded=True)
    rgb = expected[..., :3]
    display = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    png_error = float(np.max(np.abs(encoded[..., :3] - display)))
    REPORT["png16_observation"] = {
        "encoded_maximum_error": png_error,
        "rgb_tolerance": 2 / 65535,
        "opaque_alpha_maximum_error": float(np.max(np.abs(encoded[..., 3] - 1))),
    }
    np.testing.assert_allclose(encoded[..., :3], display, rtol=0, atol=2 / 65535)
    np.testing.assert_allclose(encoded[..., 3], 1, rtol=0, atol=1 / 65535)
    REPORT["render_result"] = {
        "status": "VERIFIED",
        "scope": (
            "public RenderEngine result and file APIs, generated opaque linear ramp"
        ),
        "resolution": [16, 16],
        "color": {
            "view": "Standard",
            "display": "sRGB",
            "exposure": 0,
            "gamma": 1,
            "dither_intensity": 0,
        },
        "exr_maximum_error": float(np.max(np.abs(linear - expected))),
        "png_encoded_maximum_error": float(np.max(np.abs(encoded[..., :3] - display))),
        "png_depth": 16,
        "alpha_scope": (
            "opaque RGBA only; EXR premultiplied/PNG straight alpha "
            "conversion, zero coverage and compositing NOT_RUN"
        ),
        "files": {
            label: {"path": str(path), "sha256": p1.sha(path)}
            for label, path in (("exr", exr), ("png", png))
        },
    }


def _module_path(handle):
    if sys.platform != "win32":
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.GetModuleFileNameW
    function.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    function.restype = ctypes.c_uint
    text = ctypes.create_unicode_buffer(32768)
    assert function(handle, text, len(text)), ctypes.get_last_error()
    return text.value


def _loaded_module(name):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.GetModuleHandleW
    function.argtypes, function.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    handle = function(name)
    assert handle, f"Expected loaded native module is absent: {name}"
    path = Path(_module_path(handle))
    return {"path": str(path), "sha256": p1.sha(path)}


def _native_optix(wp, device, stream, build_path):
    build = json.loads(build_path.read_text(encoding="utf-8"))
    assert build["status"] == "VERIFIED", build
    assert build["builder_sha256"] == p1.sha(ROOT / "tools/fspt/build_optix_probe.py")
    for name in ("fspt_optix_probe.cu", "fspt_optix_probe.cpp"):
        source = ROOT / "tests/acceptance/native" / name
        inputs = [
            row for path, row in build["inputs"].items() if Path(path).name == name
        ]
        assert len(inputs) == 1 and inputs[0]["sha256"] == p1.sha(source)
    files = build["artifacts"]
    for row in files.values():
        assert p1.sha(Path(row["path"])) == row["sha256"], row
    library_path, ptx_path = Path(files["library"]["path"]), Path(files["ptx"]["path"])
    library = ctypes.CDLL(str(library_path))  # The exported C ABI uses cdecl.
    assert Path(_module_path(library._handle)).resolve() == library_path.resolve()
    library.fspt_optix_probe_abi.argtypes = []
    library.fspt_optix_probe_abi.restype = ctypes.c_uint32
    assert library.fspt_optix_probe_abi() == 1

    class Stats(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("optix_version", ctypes.c_uint32),
            ("context", ctypes.c_uint64),
            ("stream", ctypes.c_uint64),
            ("gas_bytes", ctypes.c_uint64),
            ("scratch_bytes", ctypes.c_uint64),
            ("wall_ms", ctypes.c_double),
            ("retirement_result", ctypes.c_int32),
            ("cleanup_result", ctypes.c_int32),
        ]

    function = library.fspt_optix_triangle
    function.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(Stats),
    ]
    function.restype = ctypes.c_int
    ptx = ptx_path.read_bytes()
    vertices = wp.array(
        [(-0.75, -0.75, 0), (0.75, -0.75, 0), (0, 0.75, 0)],
        dtype=wp.vec3,
        device=device,
    )
    indices = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    output = wp.empty(2, dtype=wp.vec4, device=device)
    host = wp.empty(2, dtype=wp.vec4, device="cpu", pinned=True)
    producer = wp.get_stream(device)
    ready = stream.record_checked(producer)
    stream.wait_checked(ready)
    diagnostic = ctypes.create_string_buffer(8192)
    stats = Stats()

    def execute(context):
        return function(
            ptx,
            len(ptx),
            context,
            stream.cuda_stream,
            vertices.ptr,
            indices.ptr,
            output.ptr,
            diagnostic,
            len(diagnostic),
            ctypes.byref(stats),
        )

    with wp.ScopedDevice(device):
        refused = execute(
            1
        )  # An impossible expected context; rejected before any OptiX work.
        assert refused == 1 and b"CUDA context mismatch" in diagnostic.value, (
            diagnostic.value
        )
        rejection = diagnostic.value.decode()
        assert execute(device.context) == 0, diagnostic.value.decode()
        assert stats.context == device.context and stats.stream == stream.cuda_stream
        assert (
            stats.optix_version == 90000
            and stats.retirement_result == stats.cleanup_result == 0
        )
    wp.copy(host, output, stream=stream)
    copied = stream.record_checked(stream)
    wp.synchronize_event(copied)
    values = host.numpy().copy()
    expected = np.array([[1, 0.25, 0.5, 1], [-1, -1, -1, 0]], dtype=np.float32)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-6)
    raw = OUTPUT.with_name(OUTPUT.stem + "-optix-rays.npy")
    np.save(raw, values, allow_pickle=False)
    return {
        "status": "VERIFIED",
        "scope": (
            "one indexed triangle GAS and exact hit/miss rays on Warp CUDA context"
        ),
        "binding": (
            "C ABI via ctypes.CDLL in Blender Python; no CPython extension wheel"
        ),
        "library": files["library"],
        "ptx": files["ptx"],
        "build_report": {"path": str(build_path), "sha256": p1.sha(build_path)},
        "optix_driver_module": _loaded_module("nvoptix.dll"),
        "stats": {name: getattr(stats, name) for name, _type in Stats._fields_},
        "expected": expected.tolist(),
        "observed": values.tolist(),
        "negative_control": {"rejected": rejection, "before_optix_work": True},
        "raw": {"path": str(raw), "sha256": p1.sha(raw)},
        "not_qualified": [
            "FSPT optics or scene integration",
            "production performance",
            "installed redistributable",
            "GPU display sharing",
            "dynamic acceleration and solver interference",
        ],
    }


def probe_cuda(addon, optix_build):
    global wp
    runtime = p1.runtime_info(addon)
    assert runtime["warp"] == "1.19.0", runtime
    module = importlib.import_module(addon.__name__ + ".fspt.async_backend")
    wp = importlib.import_module(
        addon.__name__ + ".runtime.warp_runtime"
    )._import_warp()
    device = wp.get_device(runtime["device"])
    assert device.is_cuda and ctypes.sizeof(ctypes.c_void_p) == 8
    native = importlib.import_module(
        addon.__name__ + ".runtime.cuda_driver"
    ).cuda_driver()
    for name, args in {
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
        "cuStreamGetCtx": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "cuStreamGetFlags": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
    }.items():
        function = getattr(native.library, name)
        function.argtypes, function.restype = args, ctypes.c_int
    stream = module._nonblocking_stream(wp, device)
    retirement = stream.retirement
    try:
        with wp.ScopedDevice(device):
            context, stream_context = ctypes.c_void_p(), ctypes.c_void_p()
            flags = ctypes.c_uint()
            assert native.library.cuCtxGetCurrent(ctypes.byref(context)) == 0
            assert (
                native.library.cuStreamGetCtx(
                    stream.cuda_stream, ctypes.byref(stream_context)
                )
                == 0
            )
            assert (
                native.library.cuStreamGetFlags(stream.cuda_stream, ctypes.byref(flags))
                == 0
            )
            assert (
                context.value == stream_context.value == device.context
                and flags.value == 1
            )
        kernel = wp.Kernel(func=_capability_probe)
        output = wp.empty(1, dtype=float, device=device)
        host = wp.empty(1, dtype=float, device="cpu", pinned=True)
        wp.launch(kernel, dim=1, inputs=[output], device=device, stream=stream)
        wp.copy(host, output, stream=stream)
        event = stream.record_checked(stream)
        wp.synchronize_event(event)
        assert stream.completed(event) and float(host.numpy()[0]) == 3.0
        REPORT["cuda"] = {
            "status": "VERIFIED",
            "context": context.value,
            "stream_context": stream_context.value,
            "stream": stream.cuda_stream,
            "flags": flags.value,
            "executed_result": 3.0,
            "pinned_copy": bool(host.pinned),
            "driver_library": _loaded_module("nvcuda.dll"),
            "python_abi": {
                "version": sys.version,
                "pointer_bytes": ctypes.sizeof(ctypes.c_void_p),
            },
        }
        REPORT["optix"] = (
            _native_optix(wp, device, stream, optix_build)
            if optix_build
            else {
                "status": "NOT_RUN",
                "reason": "No explicitly bound native probe build supplied",
            }
        )
    finally:
        wp.synchronize_stream(stream)
        del stream
        gc.collect()
        assert retirement.result == 0, retirement
        REPORT.setdefault("cuda", {})["stream_destroy_result"] = retirement.result
    return runtime


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--addon-root", type=Path, default=ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    parser.add_argument("--optix-build", type=Path)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--deadline", type=float, default=120)
    parser.add_argument(
        "--demo",
        type=Path,
        action="append",
        help=(
            "Authored demo to inventory; repeat, "
            "or default to four representative demos"
        ),
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    OUTPUT = args.json_out.resolve()
    assert not OUTPUT.exists(), "A runtime report cannot overwrite an earlier attempt"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.update(
        status="RUNNING",
        verifier_sha256=p1.sha(Path(__file__)),
        context={
            "pid": os.getpid(),
            "background": bpy.app.background,
            "argv": sys.argv,
            "profile": {
                name: os.environ.get(name)
                for name in (
                    "BLENDER_USER_CONFIG",
                    "BLENDER_USER_EXTENSIONS",
                    "BLENDER_USER_SCRIPTS",
                )
            },
        },
    )
    addon = None
    registered = False
    deferred_cleanup = False
    try:
        assert bpy.app.version[:3] == (5, 2, 0) and sys.version_info[:2] == (3, 13)
        contract = json.loads(
            (ROOT / "tests/render/fixtures/fspt_acceptance_contract_v1.json").read_text(
                encoding="utf-8"
            )
        )
        assert contract["export"]["required_passes"] == [row[0] for row in PASS_SPECS]
        REPORT["apis"] = inspect_blender_apis()
        if hasattr(bpy.context.preferences.view, "show_splash"):
            bpy.context.preferences.view.show_splash = False
        p1.checkpoint(OUTPUT, REPORT, "addon_register")
        addon = p1.load_addon(args, REPORT)
        demos = args.demo or [
            ROOT / "demos" / (name + ".blend")
            for name in ("dam_break", "crown_splash", "paddle_stir", "chromatic_vortex")
        ]
        p1.checkpoint(OUTPUT, REPORT, "authored_scene_inventory")
        REPORT["authored_demos"] = inspect_authored_demos([p.resolve() for p in demos])
        p1.checkpoint(OUTPUT, REPORT, "cuda_and_native_abi")
        REPORT["runtime"] = probe_cuda(
            addon, args.optix_build.resolve() if args.optix_build else None
        )
        bpy.utils.register_class(ProbeEngine)
        registered = True
        scene = bpy.context.scene
        p1.checkpoint(OUTPUT, REPORT, "render_result_and_color")
        probe_render_result(scene)
        if args.core_only:
            REPORT["graphics"] = {"status": "NOT_RUN", "reason": "core-only invocation"}
            REPORT.update(status="VERIFIED", stage="core_complete", calls=dict(CALLS))
            p1.write_report(OUTPUT, REPORT)
            print("FSPT-P0-CORE-PASS", OUTPUT, flush=True)
            return
        assert not bpy.app.background, (
            "Complete P0 graphics probe requires a real Blender window"
        )
        window = bpy.context.window_manager.windows[0]
        area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
        region = next(r for r in area.regions if r.type == "WINDOW")
        area.spaces.active.overlay.show_overlays = False
        area.spaces.active.shading.type = "RENDERED"
        p1.checkpoint(OUTPUT, REPORT, "graphics_callbacks")
        started = time.monotonic()

        def tick():
            nonlocal registered, addon
            CALLS["timer"] += 1
            try:
                _callback_thread("timer", require_main=True)
                if REPORT.get("error"):
                    raise RuntimeError(REPORT["error"])
                if not GRAPHICS_DONE:
                    assert time.monotonic() - started < args.deadline, (
                        "No completed native graphics callback"
                    )
                    area.tag_redraw()
                    return 0.02
                assert CALLS["view_update"] > 0 and CALLS["view_draw"] > 0
                screenshot = OUTPUT.with_name(OUTPUT.stem + "-screenshot.png")
                with bpy.context.temp_override(window=window, area=area, region=region):
                    assert bpy.ops.screen.screenshot(filepath=str(screenshot)) == {
                        "FINISHED"
                    }
                REPORT["screenshot"] = {
                    "path": str(screenshot),
                    "sha256": p1.sha(screenshot),
                }
                area.spaces.active.shading.type = "SOLID"
                scene.render.engine = "BLENDER_WORKBENCH"
                bpy.utils.unregister_class(ProbeEngine)
                registered = False
                addon.unregister()
                addon = None
                REPORT.update(status="VERIFIED", stage="complete", calls=dict(CALLS))
                p1.write_report(OUTPUT, REPORT)
                print("FSPT-P0-RUNTIME-PASS", OUTPUT, flush=True)
                bpy.ops.wm.quit_blender()
                return None
            except BaseException:
                REPORT.update(
                    status="FAILED",
                    stage="graphics",
                    error=traceback.format_exc(),
                    calls=dict(CALLS),
                )
                p1.write_report(OUTPUT, REPORT)
                print("FSPT-P0-RUNTIME-FAIL", REPORT["error"], flush=True)
                os._exit(1)

        bpy.app.timers.register(tick, first_interval=0.05)
        assert bpy.app.timers.is_registered(tick)
        # The timer owns final cleanup after this script returns to Blender.
        deferred_cleanup = True
    except BaseException:
        REPORT.update(status="FAILED", error=traceback.format_exc(), calls=dict(CALLS))
        p1.write_report(OUTPUT, REPORT)
        raise
    finally:
        if registered and not deferred_cleanup:
            bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
            bpy.utils.unregister_class(ProbeEngine)
        if addon is not None and not deferred_cleanup:
            addon.unregister()


if __name__ == "__main__":
    main()
