"""Ten-thousand-frame Vulkan/CUDA PointCloud stability gate."""

import ctypes
import gc
import json
import os
from pathlib import Path

import bpy
import gpu
import warp as wp

from fluidSim_warp_blender51.blender.native_interop import NativeParticlePresenter


FRAME_COUNT = 10_000
WARMUP_FRAMES = 100
CAPACITY = 25_000


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
        ("private_usage", ctypes.c_size_t),
    ]


@wp.kernel
def animate(points: wp.array(dtype=wp.vec3), phase: float):
    index = wp.tid()
    x = index % 250
    y = index // 250
    points[index] = wp.vec3(
        (float(x) - 124.5) * 0.004,
        (float(y) - 49.5) * 0.004,
        wp.sin(phase + float(index) * 0.001) * 0.01,
    )


def process_sample():
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessHandleCount.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    process = kernel32.GetCurrentProcess()
    handles = ctypes.c_ulong()
    assert kernel32.GetProcessHandleCount(process, ctypes.byref(handles))
    memory = ProcessMemoryCountersEx()
    memory.cb = ctypes.sizeof(memory)
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    assert psapi.GetProcessMemoryInfo(
        process, ctypes.byref(memory), memory.cb)
    return {"handles": handles.value, "private_bytes": memory.private_usage}


def main():
    artifact_dir = Path(__file__).resolve(
    ).parents[1] / "docs" / "evidence" / "native_interop"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    pointcloud = bpy.data.pointclouds.new("NativeInteropStress")
    pointcloud.resize(CAPACITY)
    obj = bpy.data.objects.new("NativeInteropStress", pointcloud)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(
        bpy.context.evaluated_depsgraph_get()).data
    presenter = NativeParticlePresenter(
        CAPACITY, pointcloud=evaluated, radius=0.002)
    source = wp.empty(CAPACITY, dtype=wp.vec3, device=presenter.bridge.device)
    active_count = wp.array(
        [CAPACITY], dtype=wp.int32, device=presenter.bridge.device)
    wp.launch(
        animate,
        dim=CAPACITY,
        inputs=[source, 0.0],
        device=presenter.bridge.device,
        stream=presenter.bridge.stream,
    )
    wp.synchronize_stream(presenter.bridge.stream)
    samples = []
    baseline = None
    try:
        for frame in range(1, FRAME_COUNT + 1):
            presenter.publish_device_count(
                source, active_count, CAPACITY)
            if frame <= WARMUP_FRAMES or frame % 100 == 0:
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            if frame == WARMUP_FRAMES or frame % 1000 == 0:
                sample = {"frame": frame, **process_sample()}
                samples.append(sample)
                if baseline is None:
                    baseline = sample
        stats = dict(presenter.bridge.stats)
        slot_values = [slot.value for slot in presenter._slots]
    finally:
        presenter.close()
        bpy.data.objects.remove(obj, do_unlink=True)
        if pointcloud.users == 0:
            bpy.data.pointclouds.remove(pointcloud)
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
        gc.collect()
    final = process_sample()
    private_limit = int(baseline["private_bytes"] * 1.05)
    print("NATIVE_INTEROP_STRESS_SAMPLES", samples, final, flush=True)
    assert stats == {
        "imports": 6,
        "maps": 2,
        "published_frames": FRAME_COUNT,
        "published_bytes": FRAME_COUNT * CAPACITY * 16,
        "bulk_host_position_bytes": 0,
    }
    assert slot_values == [FRAME_COUNT // 2, FRAME_COUNT // 2]
    assert final["private_bytes"] <= private_limit, (baseline, final)
    assert final["handles"] <= baseline["handles"], (baseline, final)
    report = {
        "backend": gpu.platform.backend_type_get(),
        "device": wp.get_device("cuda:0").name,
        "frames": FRAME_COUNT,
        "warmup_frames": WARMUP_FRAMES,
        "capacity": CAPACITY,
        "validation_layer": os.environ.get("VK_INSTANCE_LAYERS", ""),
        "samples": samples,
        "baseline": baseline,
        "final": final,
        "private_bytes_limit": private_limit,
        "slot_values": slot_values,
        "bridge_stats": stats,
    }
    (artifact_dir / "stress.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_STRESS", json.dumps(report, sort_keys=True))

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        bpy.ops.wm.quit_blender()
        raise
