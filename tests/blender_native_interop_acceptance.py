"""Runtime proof that Warp-written positions draw through Blender's PointCloud path."""

from fluidSim_warp_blender51.blender.native_interop import (
    NativeParticlePresenter,
    replace_native_particle_presenter,
)
import warp as wp
from mathutils import Quaternion
import gpu
import bpy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


GRID_SIZE = 65
COUNT = GRID_SIZE * GRID_SIZE
SHRUNK_COUNT = 64
GROWN_GRID_SIZE = 80
GROWN_COUNT = GROWN_GRID_SIZE * GROWN_GRID_SIZE


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


@wp.kernel
def count_visible_points(points: wp.array(dtype=wp.vec4), result: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if points[index][3] > 0.0:
        wp.atomic_add(result, 0, 1)


@wp.kernel
def center_shrunk_points(points: wp.array(dtype=wp.vec3)):
    index = wp.tid()
    x = (float(index) - 31.5) / 31.5 * 0.65
    points[index] = wp.vec3(x, 0.0, 0.0)


@wp.kernel
def fill_grown_points(points: wp.array(dtype=wp.vec3)):
    index = wp.tid()
    x = index % GROWN_GRID_SIZE
    y = index // GROWN_GRID_SIZE
    px = (float(x) - 39.5) / 39.5 * 0.72
    py = (float(y) - 39.5) / 39.5 * 0.72
    points[index] = wp.vec3(px, py, 0.0)


def blue_particle_pixels(values, width, height):
    return sum(
        1
        for y in range(30, height - 30)
        for x in range(70, width - 110)
        for index in [(y * width + x) * 4]
        if values[index + 2] > 120 and values[index + 2] > values[index] * 2
    )


def main():
    artifact_dir = Path(__file__).resolve(
    ).parents[1] / "docs" / "evidence" / "native_interop"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for existing in list(bpy.data.objects):
        bpy.data.objects.remove(existing, do_unlink=True)
    pointcloud = bpy.data.pointclouds.new("NativeInteropPointCloud")
    pointcloud.resize(COUNT)
    obj = bpy.data.objects.new("NativeInteropPointCloud", pointcloud)
    bpy.context.scene.collection.objects.link(obj)
    obj.color = (0.08, 0.45, 1.0, 1.0)
    material = bpy.data.materials.new("NativeInteropBlue")
    material.diffuse_color = (0.08, 0.45, 1.0, 1.0)
    pointcloud.materials.append(material)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    evaluated_pointcloud = obj.evaluated_get(
        bpy.context.evaluated_depsgraph_get()).data
    print(
        "NATIVE_POINTCLOUD_POINTERS",
        {"original": pointcloud.as_pointer(),
         "evaluated": evaluated_pointcloud.as_pointer()},
        flush=True,
    )
    presenter = NativeParticlePresenter(
        COUNT, pointcloud=evaluated_pointcloud, radius=0.004)
    source = wp.empty(COUNT, dtype=wp.vec3, device=presenter.bridge.device)

    wp.launch(
        animate_points,
        dim=COUNT,
        inputs=[source, 0.35],
        device=presenter.bridge.device,
        stream=presenter.bridge.stream,
    )
    assert presenter.publish(source, COUNT)

    viewport_path = artifact_dir / "pointcloud_viewport.png"
    try:
        area = next(
            area for area in bpy.context.window.screen.areas
            if area.type == "VIEW_3D")
        region = next(region for region in area.regions if region.type == "WINDOW")
        view3d = area.spaces.active
        view3d.overlay.show_overlays = False
        view3d.shading.type = "SOLID"
        view3d.shading.light = "FLAT"
        view3d.shading.color_type = "MATERIAL"
        view3d.shading.background_type = "VIEWPORT"
        view3d.shading.background_color = (0.0, 0.0, 0.0)
        view3d.region_3d.view_location = (0.0, 0.0, 0.0)
        view3d.region_3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        view3d.region_3d.view_perspective = "ORTHO"
        with bpy.context.temp_override(area=area, region=region, space_data=view3d):
            view_result = bpy.ops.view3d.view_selected(use_all_regions=False)
            view3d.region_3d.view_distance = 1.1
            area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=5)
            result = bpy.ops.screen.screenshot(
                filepath=str(viewport_path), check_existing=False)
        assert result == {"FINISHED"}
        print(
            "NATIVE_POINTCLOUD_VIEW_DEBUG",
            {
                "objects": [
                    [item.name, item.type, item.visible_get(), item.select_get()]
                    for item in bpy.context.view_layer.objects
                ],
                "view_result": sorted(view_result),
                "view_distance": view3d.region_3d.view_distance,
                "view_location": list(view3d.region_3d.view_location),
                "perspective": view3d.region_3d.view_perspective,
            },
        )
        screenshot = bpy.data.images.load(str(viewport_path), check_existing=False)
        screenshot_width, screenshot_height = screenshot.size
        screenshot_values = [
            round(value * 255.0) for value in screenshot.pixels[:]]
        offset_x = region.x
        offset_y = region.y
        render_width = region.width
        render_height = region.height
        values = []
        for y in range(render_height):
            start = ((offset_y + y) * screenshot_width + offset_x) * 4
            values.extend(screenshot_values[start:start + render_width * 4])
        print(
            "NATIVE_POINTCLOUD_CAPTURE_DEBUG",
            {
                "area": [area.width, area.height],
                "region": [offset_x, offset_y, render_width, render_height],
                "screenshot": [screenshot_width, screenshot_height],
                "captured_values": len(values),
            },
        )
        full_colored = blue_particle_pixels(values, render_width, render_height)
        # Prime both presentation slots. The device-count shrink below must leave the inactive
        # tail visibly populated so only the GPU-written draw count can hide it.
        assert presenter.publish(source, COUNT)
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.context.view_layer.update()
        replacement_pointcloud = obj.evaluated_get(
            bpy.context.evaluated_depsgraph_get()).data
        presenter.bind_pointcloud(replacement_pointcloud)
        wp.launch(
            center_shrunk_points,
            dim=SHRUNK_COUNT,
            inputs=[source],
            device=presenter.bridge.device,
            stream=presenter.bridge.stream,
        )
        device_active_count = wp.array(
            [SHRUNK_COUNT], dtype=wp.int32, device=presenter.bridge.device)
        assert presenter.publish_device_count(source, device_active_count, COUNT)
        visible_count = wp.zeros(1, dtype=wp.int32, device=presenter.bridge.device)
        wp.launch(
            count_visible_points,
            dim=COUNT,
            inputs=[presenter.bridge.positions, visible_count],
            device=presenter.bridge.device,
            stream=presenter.bridge.stream,
        )
        wp.synchronize_stream(presenter.bridge.stream)
        visible_count_host = int(visible_count.numpy()[0])
        print("NATIVE_POINTCLOUD_VISIBLE_COUNT", visible_count_host, flush=True)
        assert visible_count_host == COUNT
        obj.hide_viewport = True
        with bpy.context.temp_override(area=area, region=region, space_data=view3d):
            area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
        obj.hide_viewport = False
        bpy.context.view_layer.update()
        replacement_pointcloud = obj.evaluated_get(
            bpy.context.evaluated_depsgraph_get()).data
        presenter.bind_pointcloud(replacement_pointcloud)
        assert presenter.publish_device_count(source, device_active_count, COUNT)
        shrunk_path = artifact_dir / "pointcloud_viewport_shrunk.png"
        with bpy.context.temp_override(area=area, region=region, space_data=view3d):
            area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=8)
            result = bpy.ops.screen.screenshot(
                filepath=str(shrunk_path), check_existing=False)
        assert result == {"FINISHED"}
        shrunk_image = bpy.data.images.load(str(shrunk_path), check_existing=False)
        shrunk_width, _shrunk_height = shrunk_image.size
        shrunk_screenshot_values = [
            round(value * 255.0) for value in shrunk_image.pixels[:]]
        shrunk_values = []
        for y in range(render_height):
            start = ((offset_y + y) * shrunk_width + offset_x) * 4
            shrunk_values.extend(
                shrunk_screenshot_values[start:start + render_width * 4])
        shrunk_colored = blue_particle_pixels(
            shrunk_values, render_width, render_height)
        assert 20 < shrunk_colored < full_colored * 0.2, (
            full_colored, shrunk_colored)
        first_generation_stats = dict(presenter.bridge.stats)
        grown_source = wp.empty(
            GROWN_COUNT, dtype=wp.vec3, device=presenter.bridge.device)
        wp.launch(
            fill_grown_points,
            dim=GROWN_COUNT,
            inputs=[grown_source],
            device=presenter.bridge.device,
            stream=presenter.bridge.stream,
        )
        wp.synchronize_stream(presenter.bridge.stream)
        grown_pointcloud_source = bpy.data.pointclouds.new(
            "NativeInteropPointCloudGrown")
        grown_pointcloud_source.resize(GROWN_COUNT)
        # Deliberately make CPU bounds wrong and outside the view. Visible GPU points below prove
        # Draw Manager culling consumes the CUDA-authored bounds instead of these CPU positions.
        grown_pointcloud_source.points.foreach_set("co", [100.0, 0.0, 0.0] * GROWN_COUNT)
        grown_pointcloud_source.update_tag()
        grown_pointcloud_source.materials.append(material)
        obj.data = grown_pointcloud_source
        bpy.context.view_layer.update()
        grown_pointcloud = obj.evaluated_get(
            bpy.context.evaluated_depsgraph_get()).data
        presenter = replace_native_particle_presenter(
            presenter,
            capacity=GROWN_COUNT,
            pointcloud=grown_pointcloud,
            radius=0.004,
            positions=grown_source,
            count=GROWN_COUNT,
        )
        grown_path = artifact_dir / "pointcloud_viewport_grown.png"
        with bpy.context.temp_override(area=area, region=region, space_data=view3d):
            area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=8)
            result = bpy.ops.screen.screenshot(
                filepath=str(grown_path), check_existing=False)
        assert result == {"FINISHED"}
        grown_image = bpy.data.images.load(str(grown_path), check_existing=False)
        grown_width, _grown_height = grown_image.size
        grown_screenshot_values = [
            round(value * 255.0) for value in grown_image.pixels[:]]
        grown_values = []
        for y in range(render_height):
            start = ((offset_y + y) * grown_width + offset_x) * 4
            grown_values.extend(
                grown_screenshot_values[start:start + render_width * 4])
        grown_colored = blue_particle_pixels(
            grown_values, render_width, render_height)
        assert grown_colored > full_colored, (full_colored, grown_colored)

        def capture_blue(path):
            with bpy.context.temp_override(
                    area=area, region=region, space_data=view3d):
                area.tag_redraw()
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=5)
                assert bpy.ops.screen.screenshot(
                    filepath=str(path), check_existing=False) == {"FINISHED"}
            image = bpy.data.images.load(str(path), check_existing=False)
            width, _height = image.size
            pixels = [round(value * 255.0) for value in image.pixels[:]]
            cropped = []
            for row in range(render_height):
                start = ((offset_y + row) * width + offset_x) * 4
                cropped.extend(pixels[start:start + render_width * 4])
            return blue_particle_pixels(cropped, render_width, render_height)

        obj.location.x = 4.0
        bpy.context.view_layer.update()
        culled_colored = capture_blue(
            artifact_dir / "pointcloud_viewport_culled.png")
        assert culled_colored < 10, culled_colored
        obj.location.x = 0.0
        bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.2), scale=(1.0, 1.0, 0.05))
        occluder = bpy.context.object
        bpy.context.view_layer.update()
        depth_colored = capture_blue(
            artifact_dir / "pointcloud_viewport_occluded.png")
        assert depth_colored < grown_colored * 0.1, (grown_colored, depth_colored)
        bpy.data.objects.remove(occluder, do_unlink=True)
        obj.select_set(False)
        assert not obj.select_get()
        obj.select_set(True)
        assert obj.select_get()
    finally:
        presenter.close()

    colored = blue_particle_pixels(values, render_width, render_height)
    assert render_width > 0 and render_height > 0
    assert len(values) == render_width * render_height * 4
    center = (render_height // 2 * render_width + render_width // 2) * 4
    center_rgba = values[center:center + 4]
    center_blue_peak = max(
        values[(y * render_width + x) * 4 + 2]
        for y in range(render_height // 2 - 2, render_height // 2 + 3)
        for x in range(render_width // 2 - 2, render_width // 2 + 3)
    )
    print(
        "NATIVE_POINTCLOUD_PIXEL_DEBUG",
        {
            "max_rgba": [max(values[channel::4]) for channel in range(4)],
            "nonzero_rgb": sum(
                1 for index in range(0, len(values), 4)
                if any(values[index + channel] for channel in range(3))
            ),
            "colored": colored,
            "center_blue_peak": center_blue_peak,
            "size": [render_width, render_height],
        },
    )
    assert colored > 3500, f"expected visible Warp points, got {colored} colored pixels"
    assert center_blue_peak > 120, "center Warp point missing"
    assert first_generation_stats == {
        "imports": 6,
        "maps": 2,
        "published_frames": 4,
        "published_bytes": COUNT * 4 * 16,
        "bulk_host_position_bytes": 0,
    }
    assert presenter.bridge.stats == {
        "imports": 6,
        "maps": 2,
        "published_frames": 1,
        "published_bytes": GROWN_COUNT * 16,
        "bulk_host_position_bytes": 0,
    }

    image_path = viewport_path
    report = {
        "backend": gpu.platform.backend_type_get(),
        "device": wp.get_device("cuda:0").name,
        "device_uuid": str(wp.get_device("cuda:0").uuid),
        "particle_count": COUNT,
        "colored_pixels": colored,
        "shrunk_particle_count": SHRUNK_COUNT,
        "shrunk_colored_pixels": shrunk_colored,
        "grown_capacity": GROWN_COUNT,
        "grown_colored_pixels": grown_colored,
        "culled_colored_pixels": culled_colored,
        "occluded_colored_pixels": depth_colored,
        "selection_roundtrip": True,
        "generation_replaced": True,
        "gpu_bounds_overrode_cpu_decoy": True,
        "center_rgba": center_rgba,
        "center_blue_peak": center_blue_peak,
        "bridge_stats": presenter.bridge.stats,
        "position_readback": False,
        "validation_layer": os.environ.get("VK_INSTANCE_LAYERS", ""),
        "draw_path": "PointCloud/DrawManager",
        "image": str(image_path),
    }
    (artifact_dir / "acceptance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_ACCEPTANCE", json.dumps(report, sort_keys=True))

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
