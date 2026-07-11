"""Runtime proof that Warp-written CUDA positions draw from a Blender Vulkan VBO."""

import json
from pathlib import Path

import bpy
import gpu
from mathutils import Matrix
import warp as wp

from fluidSim_warp_blender51.blender.native_interop import NativeParticlePresenter


WIDTH = 256
HEIGHT = 256
GRID_SIZE = 65
COUNT = GRID_SIZE * GRID_SIZE


@wp.kernel
def animate_points(points: wp.array(dtype=wp.vec3), phase: float):
    index = wp.tid()
    x = index % GRID_SIZE
    y = index // GRID_SIZE
    px = (float(x) - 32.0) / 32.0 * 0.72
    py = (float(y) - 32.0) / 32.0 * 0.72
    wave = 0.035 * wp.sin(phase + px * 8.0)
    points[index] = wp.vec3(px, py + wave, 0.0)
    if index == COUNT - 1:
        points[index] = wp.vec3(0.0, 0.0, 0.0)


def save_pixels(path, buffer):
    image = bpy.data.images.new("NativeInteropProof", WIDTH, HEIGHT)
    buffer.dimensions = WIDTH * HEIGHT * 4
    image.pixels = [value / 255.0 for value in buffer]
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def main():
    artifact_dir = Path(__file__).resolve(
    ).parents[1] / "docs" / "evidence" / "native_interop"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    presenter = NativeParticlePresenter(COUNT, point_size=18.0)
    source = wp.empty(COUNT, dtype=wp.vec3, device=presenter.bridge.device)

    wp.launch(
        animate_points,
        dim=COUNT,
        inputs=[source, 0.35],
        device=presenter.bridge.device,
        stream=presenter.bridge.stream,
    )
    assert presenter.publish(source, COUNT)

    offscreen = gpu.types.GPUOffScreen(WIDTH, HEIGHT)
    try:
        with offscreen.bind():
            framebuffer = gpu.state.active_framebuffer_get()
            framebuffer.clear(color=(0.0, 0.0, 0.0, 1.0))
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                presenter.draw()
            pixels = framebuffer.read_color(0, 0, WIDTH, HEIGHT, 4, 0, "UBYTE")
    finally:
        offscreen.free()
        presenter.close()

    pixels.dimensions = WIDTH * HEIGHT * 4
    values = list(pixels)
    colored = sum(
        1
        for index in range(0, len(values), 4)
        if values[index + 2] > 120 and values[index + 2] > values[index] * 2
    )
    center = (HEIGHT // 2 * WIDTH + WIDTH // 2) * 4
    center_rgba = values[center:center + 4]
    center_blue_peak = max(
        values[(y * WIDTH + x) * 4 + 2]
        for y in range(HEIGHT // 2 - 2, HEIGHT // 2 + 3)
        for x in range(WIDTH // 2 - 2, WIDTH // 2 + 3)
    )
    assert colored > 3500, f"expected visible Warp points, got {colored} colored pixels"
    assert center_blue_peak > 120, "center Warp point missing"
    assert presenter.bridge.stats == {
        "imports": 3,
        "maps": 1,
        "published_frames": 1,
        "published_bytes": COUNT * 12,
        "bulk_host_position_bytes": 0,
    }

    image_path = artifact_dir / "warp_cuda_vulkan_points.png"
    save_pixels(image_path, pixels)
    report = {
        "backend": gpu.platform.backend_type_get(),
        "device": wp.get_device("cuda:0").name,
        "device_uuid": str(wp.get_device("cuda:0").uuid),
        "particle_count": COUNT,
        "colored_pixels": colored,
        "center_rgba": center_rgba,
        "center_blue_peak": center_blue_peak,
        "bridge_stats": presenter.bridge.stats,
        "position_readback": False,
        "image": str(image_path),
    }
    (artifact_dir / "acceptance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_ACCEPTANCE", json.dumps(report, sort_keys=True))
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
