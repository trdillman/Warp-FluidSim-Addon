"""Vulkan/CUDA particle presentation without CPU position materialization."""

from __future__ import annotations

import ctypes


class NativeInteropUnavailable(RuntimeError):
    """Raised when the custom Blender Vulkan interop surface is unavailable."""


_DEFERRED_SLOT_CLEANUP = []


def _close_slots_with_retry(slots, primary_error=None):
    for slot in list(slots):
        last_error = None
        for _attempt in range(2):
            try:
                slot.close()
            except Exception as error:
                last_error = error
                if primary_error is not None:
                    primary_error.add_note(str(error))
            else:
                last_error = None
                break
        if last_error is not None:
            _DEFERRED_SLOT_CLEANUP.append(slot)


def _retry_deferred_slot_cleanup():
    pending = list(_DEFERRED_SLOT_CLEANUP)
    _DEFERRED_SLOT_CLEANUP.clear()
    _close_slots_with_retry(pending)


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _Win32Handle),
        ("object", ctypes.c_void_p),
    ]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalSemaphoreHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _FenceValue(ctypes.Structure):
    _fields_ = [("value", ctypes.c_ulonglong)]


class _NvSciSync(ctypes.Union):
    _fields_ = [("fence", ctypes.c_void_p), ("reserved", ctypes.c_ulonglong)]


class _SignalKeyedMutex(ctypes.Structure):
    _fields_ = [("key", ctypes.c_ulonglong)]


class _WaitKeyedMutex(ctypes.Structure):
    _fields_ = [("key", ctypes.c_ulonglong), ("timeout_ms", ctypes.c_uint)]


class _SignalInner(ctypes.Structure):
    _fields_ = [
        ("fence", _FenceValue),
        ("nv_sci_sync", _NvSciSync),
        ("keyed_mutex", _SignalKeyedMutex),
        ("reserved", ctypes.c_uint * 12),
    ]


class _WaitInner(ctypes.Structure):
    _fields_ = [
        ("fence", _FenceValue),
        ("nv_sci_sync", _NvSciSync),
        ("keyed_mutex", _WaitKeyedMutex),
        ("reserved", ctypes.c_uint * 10),
    ]


class _ExternalSemaphoreSignalParams(ctypes.Structure):
    _fields_ = [
        ("params", _SignalInner),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalSemaphoreWaitParams(ctypes.Structure):
    _fields_ = [
        ("params", _WaitInner),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


def _uuid_bytes(value: str) -> bytes:
    text = value.removeprefix("GPU-").replace("-", "")
    result = bytes.fromhex(text)
    if len(result) != 16:
        raise NativeInteropUnavailable(f"invalid CUDA UUID: {value!r}")
    return result


def _make_position_radius_kernel(wp):
    def _publish(source, target, radius, step):
        index = wp.tid()
        position = source[index * step]
        target[index] = wp.vec4(
            position[0], position[1], position[2], radius)

    _publish.__annotations__ = {
        "source": wp.array(dtype=wp.vec3),
        "target": wp.array(dtype=wp.vec4),
        "radius": float,
        "step": int,
    }
    return wp.kernel(_publish)


def _make_counted_position_radius_kernel(wp):
    def _publish(source, active_count, target, radius, step):
        index = wp.tid()
        source_index = index * step
        if source_index < active_count[0]:
            position = source[source_index]
            target[index] = wp.vec4(
                position[0], position[1], position[2], radius)

    _publish.__annotations__ = {
        "source": wp.array(dtype=wp.vec3),
        "active_count": wp.array(dtype=wp.int32),
        "target": wp.array(dtype=wp.vec4),
        "radius": float,
        "step": int,
    }
    return wp.kernel(_publish)


def _make_draw_vertex_count_kernel(wp):
    def _write(count, draw_command):
        draw_command[0] = count * 8
        draw_command[1] = 1
        draw_command[2] = 0
        draw_command[3] = 0

    _write.__annotations__ = {
        "count": int,
        "draw_command": wp.array(dtype=wp.int32),
    }
    return wp.kernel(_write)


def _make_device_count_draw_vertex_count_kernel(wp):
    def _write(active_count, draw_command):
        count = active_count[0]
        if count > 0:
            draw_command[0] = count * 8
        else:
            draw_command[0] = 0
        draw_command[1] = 1
        draw_command[2] = 0
        draw_command[3] = 0

    _write.__annotations__ = {
        "active_count": wp.array(dtype=wp.int32),
        "draw_command": wp.array(dtype=wp.int32),
    }
    return wp.kernel(_write)


def _make_bounds_init_kernel(wp):
    def _init(bounds):
        bounds[0] = 3.402823e38
        bounds[1] = 3.402823e38
        bounds[2] = 3.402823e38
        bounds[3] = -3.402823e38
        bounds[4] = -3.402823e38
        bounds[5] = -3.402823e38

    _init.__annotations__ = {"bounds": wp.array(dtype=float)}
    return wp.kernel(_init)


def _make_bounds_reduce_kernel(wp):
    def _reduce(positions, draw_command, bounds):
        index = wp.tid()
        if index * 8 < draw_command[0]:
            position = positions[index]
            wp.atomic_min(bounds, 0, position[0])
            wp.atomic_min(bounds, 1, position[1])
            wp.atomic_min(bounds, 2, position[2])
            wp.atomic_max(bounds, 3, position[0])
            wp.atomic_max(bounds, 4, position[1])
            wp.atomic_max(bounds, 5, position[2])

    _reduce.__annotations__ = {
        "positions": wp.array(dtype=wp.vec4),
        "draw_command": wp.array(dtype=wp.int32),
        "bounds": wp.array(dtype=float),
    }
    return wp.kernel(_reduce)


def _make_bounds_encode_kernel(wp):
    def _encode(bounds, draw_command, target, radius):
        if draw_command[0] > 0:
            min_x = bounds[0] - radius
            min_y = bounds[1] - radius
            min_z = bounds[2] - radius
            max_x = bounds[3] + radius
            max_y = bounds[4] + radius
            max_z = bounds[5] + radius
            target[0] = min_x
            target[1] = min_y
            target[2] = min_z
            target[4] = max_x
            target[5] = min_y
            target[6] = min_z
            target[8] = min_x
            target[9] = max_y
            target[10] = min_z
            target[12] = min_x
            target[13] = min_y
            target[14] = max_z
        else:
            target[0] = 0.0
            target[1] = 0.0
            target[2] = 0.0
            target[4] = 0.0
            target[5] = 0.0
            target[6] = 0.0
            target[8] = 0.0
            target[9] = 0.0
            target[10] = 0.0
            target[12] = 0.0
            target[13] = 0.0
            target[14] = 0.0
        target[3] = 0.0
        target[7] = 0.0
        target[11] = 0.0
        target[15] = 0.0
        target[16] = 0.0
        target[17] = 0.0
        target[18] = 0.0
        target[19] = 0.0

    _encode.__annotations__ = {
        "bounds": wp.array(dtype=float),
        "draw_command": wp.array(dtype=wp.int32),
        "target": wp.array(dtype=float),
        "radius": float,
    }
    return wp.kernel(_encode)


def _make_clear_position_radius_kernel(wp):
    def _clear(target, start):
        index = wp.tid() + start
        target[index] = wp.vec4(0.0, 0.0, 0.0, 0.0)

    _clear.__annotations__ = {
        "target": wp.array(dtype=wp.vec4),
        "start": int,
    }
    return wp.kernel(_clear)


def _make_height_sample_kernel(wp):
    def _sample(source, target, step):
        index = wp.tid()
        target[index] = source[index * step][2]

    _sample.__annotations__ = {
        "source": wp.array(dtype=wp.vec3),
        "target": wp.array(dtype=float),
        "step": int,
    }
    return wp.kernel(_sample)


def _make_speed_sample_kernel(wp):
    def _sample(source, target, step):
        index = wp.tid()
        velocity = source[index * step]
        target[index] = wp.length(velocity)

    _sample.__annotations__ = {
        "source": wp.array(dtype=wp.vec3),
        "target": wp.array(dtype=float),
        "step": int,
    }
    return wp.kernel(_sample)


def _make_scalar_sample_kernel(wp):
    def _sample(source, target, step):
        index = wp.tid()
        target[index] = source[index * step]

    _sample.__annotations__ = {
        "source": wp.array(dtype=float),
        "target": wp.array(dtype=float),
        "step": int,
    }
    return wp.kernel(_sample)


class _CudaDriver:
    _MEMORY_OPAQUE_WIN32 = 2
    _TIMELINE_WIN32 = 10

    def __init__(self):
        self.lib = ctypes.WinDLL("nvcuda.dll")
        self.ctx_push = self._fn(
            "cuCtxPushCurrent_v2", ctypes.c_int, ctypes.c_void_p)
        self.ctx_pop = self._fn("cuCtxPopCurrent_v2", ctypes.c_int,
                                ctypes.POINTER(ctypes.c_void_p))
        self.import_memory = self._fn(
            "cuImportExternalMemory", ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalMemoryHandleDesc),
        )
        self.map_memory = self._fn(
            "cuExternalMemoryGetMappedBuffer", ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p,
            ctypes.POINTER(_ExternalMemoryBufferDesc),
        )
        self.free_memory = self._fn("cuMemFree_v2", ctypes.c_int,
                                    ctypes.c_ulonglong)
        self.destroy_memory = self._fn("cuDestroyExternalMemory", ctypes.c_int,
                                       ctypes.c_void_p)
        self.import_semaphore = self._fn(
            "cuImportExternalSemaphore", ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalSemaphoreHandleDesc),
        )
        self.destroy_semaphore = self._fn(
            "cuDestroyExternalSemaphore", ctypes.c_int, ctypes.c_void_p)
        self.wait_semaphore = self._fn(
            "cuWaitExternalSemaphoresAsync", ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalSemaphoreWaitParams),
            ctypes.c_uint, ctypes.c_void_p,
        )
        self.signal_semaphore = self._fn(
            "cuSignalExternalSemaphoresAsync", ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalSemaphoreSignalParams),
            ctypes.c_uint, ctypes.c_void_p,
        )

    def _fn(self, name, restype, *argtypes):
        fn = getattr(self.lib, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)
        return fn

    @staticmethod
    def check(result, operation):
        if result != 0:
            raise NativeInteropUnavailable(
                f"CUDA driver operation {operation} failed with code {result}")

    def push(self, context):
        self.check(self.ctx_push(ctypes.c_void_p(context)), "cuCtxPushCurrent")

    def pop(self):
        previous = ctypes.c_void_p()
        self.check(self.ctx_pop(ctypes.byref(previous)), "cuCtxPopCurrent")

    def import_external_memory(self, handle, size):
        desc = _ExternalMemoryHandleDesc()
        desc.type = self._MEMORY_OPAQUE_WIN32
        desc.handle.win32.handle = handle
        desc.size = size
        memory = ctypes.c_void_p()
        self.check(self.import_memory(ctypes.byref(memory), ctypes.byref(desc)),
                   "cuImportExternalMemory")
        mapped = ctypes.c_ulonglong()
        buffer_desc = _ExternalMemoryBufferDesc(size=size)
        try:
            self.check(self.map_memory(ctypes.byref(mapped), memory,
                                       ctypes.byref(buffer_desc)),
                       "cuExternalMemoryGetMappedBuffer")
        except Exception as error:
            try:
                self.check(self.destroy_memory(memory),
                           "cuDestroyExternalMemory")
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            raise
        return memory, mapped.value

    def import_timeline(self, handle):
        desc = _ExternalSemaphoreHandleDesc()
        desc.type = self._TIMELINE_WIN32
        desc.handle.win32.handle = handle
        semaphore = ctypes.c_void_p()
        self.check(self.import_semaphore(ctypes.byref(semaphore),
                                         ctypes.byref(desc)),
                   "cuImportExternalSemaphore")
        return semaphore

    def wait(self, semaphore, value, stream):
        params = _ExternalSemaphoreWaitParams()
        params.params.fence.value = value
        semaphores = (ctypes.c_void_p * 1)(semaphore.value)
        self.check(self.wait_semaphore(semaphores, ctypes.byref(params), 1,
                                       ctypes.c_void_p(stream)),
                   "cuWaitExternalSemaphoresAsync")

    def signal(self, semaphore, value, stream):
        params = _ExternalSemaphoreSignalParams()
        params.params.fence.value = value
        semaphores = (ctypes.c_void_p * 1)(semaphore.value)
        self.check(self.signal_semaphore(semaphores, ctypes.byref(params), 1,
                                         ctypes.c_void_p(stream)),
                   "cuSignalExternalSemaphoresAsync")


def _validate_external_descriptor(descriptor, capacity):
    required = {
        "api_version",
        "generation",
        "backend",
        "memory_handle_type",
        "semaphore_handle_type",
        "memory_handle",
        "memory_size",
        "draw_vertex_count_offset",
        "pointcloud_bounds_offset",
        "release_semaphore_handle",
        "completion_semaphore_handle",
        "device_uuid",
        "device_luid",
        "device_node_mask",
        "memory_exportable",
        "memory_importable",
        "memory_dedicated_only",
        "semaphore_exportable",
        "semaphore_importable",
        "max_timeline_value_difference",
    }
    missing = sorted(required.difference(descriptor))
    if missing:
        raise NativeInteropUnavailable(
            f"Blender external descriptor is missing {', '.join(missing)}")
    if descriptor["api_version"] != 1:
        raise NativeInteropUnavailable(
            f"unsupported Blender external API version {descriptor['api_version']!r}")
    if descriptor["backend"] != "VULKAN":
        raise NativeInteropUnavailable(
            "native interop requires Blender's Vulkan backend")
    if descriptor["memory_handle_type"] != "OPAQUE_WIN32":
        raise NativeInteropUnavailable(
            "unsupported Blender external-memory handle type")
    if descriptor["semaphore_handle_type"] != "OPAQUE_WIN32":
        raise NativeInteropUnavailable(
            "unsupported Blender external-semaphore handle type")
    if not all((
            descriptor["memory_exportable"],
            descriptor["memory_importable"],
            descriptor["semaphore_exportable"],
            descriptor["semaphore_importable"],
            descriptor["max_timeline_value_difference"] > 0,
    )):
        raise NativeInteropUnavailable(
            "required Vulkan external capabilities are unavailable")
    if descriptor["generation"] <= 0:
        raise NativeInteropUnavailable(
            "invalid Blender external lease generation")
    if len(descriptor["device_uuid"]) != 16:
        raise NativeInteropUnavailable("invalid Blender Vulkan device UUID")
    if len(descriptor["device_luid"]) not in {0, 8}:
        raise NativeInteropUnavailable("invalid Blender Vulkan device LUID")
    if descriptor["device_node_mask"] < 0:
        raise NativeInteropUnavailable(
            "invalid Blender Vulkan device node mask")
    if min(
            descriptor["memory_handle"],
            descriptor["release_semaphore_handle"],
            descriptor["completion_semaphore_handle"],
    ) <= 0:
        raise NativeInteropUnavailable("invalid Blender Win32 interop handle")

    memory_size = int(descriptor["memory_size"])
    draw_offset = int(descriptor["draw_vertex_count_offset"])
    bounds_offset = int(descriptor["pointcloud_bounds_offset"])
    if memory_size < int(capacity) * 16:
        raise NativeInteropUnavailable(
            "Blender external allocation is smaller than capacity")
    if draw_offset < int(capacity) * 16 or draw_offset + 16 > memory_size:
        raise NativeInteropUnavailable(
            "invalid Blender draw-count metadata range")
    if bounds_offset < draw_offset + 16 or bounds_offset + 80 > memory_size:
        raise NativeInteropUnavailable(
            "invalid Blender point-cloud bounds metadata range")


class CudaExternalBuffer:
    """One persistently imported Blender VBO and two directional timelines."""

    def __init__(self, descriptor, capacity):
        _validate_external_descriptor(descriptor, capacity)
        import warp as wp

        self.wp = wp
        self.device = wp.get_device("cuda:0")
        if _uuid_bytes(self.device.uuid) != descriptor["device_uuid"]:
            raise NativeInteropUnavailable(
                "Blender Vulkan and Warp CUDA device UUIDs do not match")
        self.driver = _CudaDriver()
        self.context = self.device.context
        self.stream = self.device.stream
        self.memory = ctypes.c_void_p()
        self.release = ctypes.c_void_p()
        self.completion = ctypes.c_void_p()
        self.mapped_pointer = 0
        self.context_pushed = False
        self.closed = False
        self.capacity = int(capacity)
        self.generation = int(descriptor["generation"])
        self._publication_profile_events = None
        self.publication_profile = None
        self.stats = {
            "imports": 0,
            "maps": 0,
            "published_frames": 0,
            "published_bytes": 0,
            "bulk_host_position_bytes": 0,
        }

        try:
            self.driver.push(self.context)
            self.context_pushed = True
            try:
                self.memory, self.mapped_pointer = self.driver.import_external_memory(
                    descriptor["memory_handle"], descriptor["memory_size"])
                self.release = self.driver.import_timeline(
                    descriptor["release_semaphore_handle"])
                self.completion = self.driver.import_timeline(
                    descriptor["completion_semaphore_handle"])
            finally:
                self.driver.pop()
                self.context_pushed = False
            self.positions = wp.array(
                ptr=self.mapped_pointer,
                capacity=descriptor["memory_size"],
                shape=(capacity,),
                dtype=wp.vec4,
                device=self.device,
            )
            draw_offset = int(descriptor["draw_vertex_count_offset"])
            self.draw_vertex_count = wp.array(
                ptr=self.mapped_pointer + draw_offset,
                capacity=16,
                shape=(4,),
                dtype=wp.int32,
                device=self.device,
            )
            bounds_offset = int(descriptor["pointcloud_bounds_offset"])
            self.pointcloud_bounds = wp.array(
                ptr=self.mapped_pointer + bounds_offset,
                capacity=80,
                shape=(20,),
                dtype=float,
                device=self.device,
            )
            self.bounds_min_max = wp.empty(6, dtype=float, device=self.device)
            self.publish_kernel = _make_position_radius_kernel(wp)
            self.counted_publish_kernel = _make_counted_position_radius_kernel(
                wp)
            self.clear_kernel = _make_clear_position_radius_kernel(wp)
            self.draw_vertex_count_kernel = _make_draw_vertex_count_kernel(wp)
            self.device_count_draw_kernel = _make_device_count_draw_vertex_count_kernel(
                wp)
            self.bounds_init_kernel = _make_bounds_init_kernel(wp)
            self.bounds_reduce_kernel = _make_bounds_reduce_kernel(wp)
            self.bounds_encode_kernel = _make_bounds_encode_kernel(wp)
            self.visible_count = None
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
                error._native_interop_bridge = self
            raise
        self.stats["imports"] = 3
        self.stats["maps"] = 1

    def publication_profiling_enable(self):
        """Enable synchronous CUDA event timing for explicit benchmark runs."""
        if self._publication_profile_events is None:
            self._publication_profile_events = tuple(
                self.wp.Event(self.device, enable_timing=True) for _index in range(5))

    def _publication_profile_collect(self):
        start, waited, packed, metadata, completed = self._publication_profile_events
        elapsed = self.wp.get_event_elapsed_time
        self.publication_profile = {
            "external_wait_ms": elapsed(start, waited, synchronize=True),
            "position_pack_ms": elapsed(waited, packed, synchronize=True),
            "metadata_ms": elapsed(packed, metadata, synchronize=True),
            "cuda_publication_ms": elapsed(start, completed, synchronize=True),
        }

    def publish(self, source, count, value, radius, step=1):
        clear_end = self.capacity if self.visible_count is None else self.visible_count
        clear_count = max(0, clear_end - count)
        profile = getattr(self, "_publication_profile_events", None)
        self.driver.push(self.context)
        self.context_pushed = True
        try:
            if profile is not None:
                self.stream.record_event(profile[0])
            self.driver.wait(self.release, value, self.stream.cuda_stream)
            if profile is not None:
                self.stream.record_event(profile[1])
            if count > 0:
                self.wp.launch(
                    self.publish_kernel,
                    dim=count,
                    inputs=[source, self.positions, float(radius), int(step)],
                    device=self.device,
                    stream=self.stream,
                )
            if profile is not None:
                self.stream.record_event(profile[2])
            if clear_count > 0:
                self.wp.launch(
                    self.clear_kernel,
                    dim=clear_count,
                    inputs=[self.positions, int(count)],
                    device=self.device,
                    stream=self.stream,
                )
            self.wp.launch(
                self.draw_vertex_count_kernel,
                dim=1,
                inputs=[int(count), self.draw_vertex_count],
                device=self.device,
                stream=self.stream,
            )
            self._publish_bounds(count, radius)
            if profile is not None:
                self.stream.record_event(profile[3])
            self.driver.signal(self.completion, value, self.stream.cuda_stream)
            if profile is not None:
                self.stream.record_event(profile[4])
        finally:
            self.driver.pop()
            self.context_pushed = False
        self.visible_count = int(count)
        self.stats["published_frames"] += 1
        self.stats["published_bytes"] += (count + clear_count) * 16

    def publish_device_count(
            self,
            source,
            active_count,
            output_count,
            value,
            radius,
            step=1,
            repack=True):
        profile = getattr(self, "_publication_profile_events", None)
        self.driver.push(self.context)
        self.context_pushed = True
        try:
            if profile is not None:
                self.stream.record_event(profile[0])
            self.driver.wait(self.release, value, self.stream.cuda_stream)
            if profile is not None:
                self.stream.record_event(profile[1])
            if repack:
                self.wp.launch(
                    self.counted_publish_kernel,
                    dim=output_count,
                    inputs=[
                        source,
                        active_count,
                        self.positions,
                        float(radius),
                        int(step),
                    ],
                    device=self.device,
                    stream=self.stream,
                )
            if profile is not None:
                self.stream.record_event(profile[2])
            self.wp.launch(
                self.device_count_draw_kernel,
                dim=1,
                inputs=[active_count, self.draw_vertex_count],
                device=self.device,
                stream=self.stream,
            )
            self._publish_bounds(output_count, radius)
            if profile is not None:
                self.stream.record_event(profile[3])
            self.driver.signal(self.completion, value, self.stream.cuda_stream)
            if profile is not None:
                self.stream.record_event(profile[4])
        finally:
            self.driver.pop()
            self.context_pushed = False
        self.visible_count = int(output_count)
        self.stats["published_frames"] += 1
        self.stats["published_bytes"] += output_count * 16 if repack else 0

    def _publish_bounds(self, output_count, radius):
        self.wp.launch(
            self.bounds_init_kernel,
            dim=1,
            inputs=[self.bounds_min_max],
            device=self.device,
            stream=self.stream,
        )
        if output_count > 0:
            self.wp.launch(
                self.bounds_reduce_kernel,
                dim=output_count,
                inputs=[self.positions, self.draw_vertex_count,
                        self.bounds_min_max],
                device=self.device,
                stream=self.stream,
            )
        self.wp.launch(
            self.bounds_encode_kernel,
            dim=1,
            inputs=[
                self.bounds_min_max,
                self.draw_vertex_count,
                self.pointcloud_bounds,
                float(radius),
            ],
            device=self.device,
            stream=self.stream,
        )

    def close(self):
        if self.closed:
            return
        try:
            self.wp.synchronize_stream(self.stream)
        except Exception:
            # CUDA imports must remain intact while queued work may still use them.
            raise

        errors = []

        context_ready = getattr(self, "context_pushed", False)
        if not context_ready:
            try:
                self.driver.push(self.context)
                self.context_pushed = True
                context_ready = True
            except Exception as error:
                errors.append(error)

        if context_ready:
            for name in ("completion", "release"):
                semaphore = getattr(self, name)
                if not semaphore.value:
                    continue
                try:
                    self.driver.check(
                        self.driver.destroy_semaphore(semaphore),
                        "cuDestroyExternalSemaphore",
                    )
                except Exception as error:
                    errors.append(error)
                else:
                    setattr(self, name, ctypes.c_void_p())
            if self.mapped_pointer:
                try:
                    self.driver.check(
                        self.driver.free_memory(self.mapped_pointer),
                        "cuMemFree",
                    )
                except Exception as error:
                    errors.append(error)
                else:
                    self.mapped_pointer = 0
            if self.memory.value and not self.mapped_pointer:
                try:
                    self.driver.check(
                        self.driver.destroy_memory(self.memory),
                        "cuDestroyExternalMemory",
                    )
                except Exception as error:
                    errors.append(error)
                else:
                    self.memory = ctypes.c_void_p()
            try:
                self.driver.pop()
                self.context_pushed = False
            except Exception as error:
                errors.append(error)

        self.closed = (
            not self.completion.value
            and not self.release.value
            and not self.mapped_pointer
            and not self.memory.value
            and not getattr(self, "context_pushed", False)
        )
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(str(error))
            raise primary


class _NativeParticleSlot:
    """One independently synchronized PointCloud presentation allocation."""

    def __init__(self, capacity):
        import gpu

        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type="F32", len=4, fetch_mode="FLOAT")
        self.vbo = gpu.types.GPUVertBuf(
            format=fmt, len=capacity, external=True)
        self.lease = self.vbo.external_lease_acquire()
        self.bridge = None
        self.pointcloud = None
        try:
            self.bridge = CudaExternalBuffer(
                self.lease.descriptor_get(), capacity)
        except Exception as error:
            self.bridge = getattr(error, "_native_interop_bridge", None)
            if self.bridge is None:
                try:
                    self.lease.close()
                except Exception as cleanup_error:
                    error.add_note(str(cleanup_error))
                    _DEFERRED_SLOT_CLEANUP.append(self)
                else:
                    self.lease = None
                    self.vbo = None
            else:
                _close_slots_with_retry([self], error)
            raise
        self.active_count = 0
        self.value = 0

    def bind_pointcloud(self, pointcloud):
        self.vbo._pointcloud_position_bind(pointcloud)
        self.pointcloud = pointcloud

    def publish(self, positions, count, radius, step=1):
        bytes_before = self.bridge.stats["published_bytes"]
        self.value += 1
        self.lease.release(self.value)
        self.bridge.publish(positions, count, self.value, radius, step)
        self.lease.acquire(self.value)
        self.vbo._pointcloud_position_count_set(count)
        self.lease.submit(self.value)
        if getattr(self.bridge, "_publication_profile_events", None) is not None:
            self.bridge._publication_profile_collect()
        self.active_count = int(count)
        return self.bridge.stats["published_bytes"] - bytes_before

    def publish_device_count(
            self,
            positions,
            active_count,
            output_count,
            radius,
            step=1,
            repack=True):
        bytes_before = self.bridge.stats["published_bytes"]
        self.value += 1
        self.lease.release(self.value)
        self.bridge.publish_device_count(
            positions,
            active_count,
            output_count,
            self.value,
            radius,
            step,
            repack,
        )
        self.lease.acquire(self.value)
        self.vbo._pointcloud_position_count_set(output_count)
        self.lease.submit(self.value)
        if getattr(self.bridge, "_publication_profile_events", None) is not None:
            self.bridge._publication_profile_collect()
        self.active_count = int(output_count)
        return self.bridge.stats["published_bytes"] - bytes_before

    def close(self):
        errors = []
        if self.vbo is not None:
            try:
                self.vbo._pointcloud_position_unbind()
            except Exception as error:
                errors.append(error)
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception as error:
                errors.append(error)
            else:
                self.bridge = None
        if self.bridge is None and self.lease is not None:
            try:
                self.lease.close()
            except Exception as error:
                errors.append(error)
            else:
                self.lease = None
        if not errors:
            self.vbo = None
            self.pointcloud = None
        else:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(str(error))
            raise primary


class _PresenterBridgeView:
    """Compatibility and aggregate diagnostics for presenter slots."""

    def __init__(self, slots):
        first = slots[0].bridge
        self.device = first.device
        self.stream = first.stream
        self.wp = first.wp
        self.positions = first.positions
        self.stats = {
            "imports": sum(slot.bridge.stats["imports"] for slot in slots),
            "maps": sum(slot.bridge.stats["maps"] for slot in slots),
            "published_frames": 0,
            "published_bytes": 0,
            "bulk_host_position_bytes": 0,
        }


class NativeParticlePresenter:
    """Publish Warp particles through two persistent Blender PointCloud VBOs."""

    def __init__(self, capacity, pointcloud, radius, profile_publication=False):
        import gpu

        _retry_deferred_slot_cleanup()
        if not hasattr(gpu.types.GPUVertBuf, "_pointcloud_position_bind"):
            raise NativeInteropUnavailable(
                "Blender does not expose native PointCloud interop")
        self._slots = []
        try:
            for _index in range(2):
                self._slots.append(_NativeParticleSlot(capacity))
            if profile_publication:
                for slot in self._slots:
                    slot.bridge.publication_profiling_enable()
            self._slot_index = -1
            self.pointcloud = pointcloud
            self.radius = float(radius)
            self.active_count = 0
            self.bridge = _PresenterBridgeView(self._slots)
            self.capacity = int(capacity)
            self._display_values = None
            self._display_kernels = None
            self._materialization_cache_key = None
            self._materialization_cache_values = None
            self.materialization_stats = {"reads": 0, "bytes": 0, "reason": ""}
            self._slots[0].bind_pointcloud(pointcloud)
            self._slots[0].vbo._pointcloud_position_count_set(0)
        except Exception as error:
            _close_slots_with_retry(self._slots, error)
            raise

    @property
    def vbo(self):
        return self._slots[max(self._slot_index, 0)].vbo

    def bind_pointcloud(self, pointcloud):
        self.pointcloud = pointcloud

    def _restore_active_binding(self):
        slot = self._slots[max(self._slot_index, 0)]
        slot.bind_pointcloud(self.pointcloud)
        slot.vbo._pointcloud_position_count_set(self.active_count)

    def publish(self, positions, count, step=1):
        if count <= 0 and self.active_count <= 0:
            self.active_count = 0
            self.vbo._pointcloud_position_count_set(0)
            return True
        next_slot_index = (self._slot_index + 1) % len(self._slots)
        previous_slot = self._slots[self._slot_index] if self._slot_index >= 0 else None
        slot = self._slots[next_slot_index]
        try:
            slot.bind_pointcloud(self.pointcloud)
            published_bytes = slot.publish(positions, count, self.radius, step)
        except Exception as error:
            try:
                slot.vbo._pointcloud_position_unbind()
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            if previous_slot is not None and previous_slot is not slot:
                try:
                    previous_slot.bind_pointcloud(self.pointcloud)
                except Exception as cleanup_error:
                    error.add_note(str(cleanup_error))
            raise
        self._slot_index = next_slot_index
        self.active_count = int(count)
        self.bridge.positions = slot.bridge.positions
        self.bridge.stream = slot.bridge.stream
        self.bridge.stats["published_frames"] += 1
        self.bridge.stats["published_bytes"] += published_bytes or 0
        return True

    def publish_device_count(
            self,
            positions,
            active_count,
            output_count,
            step=1,
            repack=True):
        next_slot_index = (self._slot_index + 1) % len(self._slots)
        previous_slot = self._slots[self._slot_index] if self._slot_index >= 0 else None
        slot = self._slots[next_slot_index]
        try:
            slot.bind_pointcloud(self.pointcloud)
            published_bytes = slot.publish_device_count(
                positions,
                active_count,
                output_count,
                self.radius,
                step,
                repack,
            )
        except Exception as error:
            try:
                slot.vbo._pointcloud_position_unbind()
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            if previous_slot is not None and previous_slot is not slot:
                try:
                    previous_slot.bind_pointcloud(self.pointcloud)
                except Exception as cleanup_error:
                    error.add_note(str(cleanup_error))
            raise
        self._slot_index = next_slot_index
        self.active_count = int(output_count)
        self.bridge.positions = slot.bridge.positions
        self.bridge.stream = slot.bridge.stream
        self.bridge.stats["published_frames"] += 1
        self.bridge.stats["published_bytes"] += published_bytes or 0
        return True

    @property
    def publication_profile(self):
        return self._slots[max(self._slot_index, 0)].bridge.publication_profile

    def publication_profiling_enable(self):
        """Enable synchronized CUDA event timing for explicit benchmark runs."""
        for slot in self._slots:
            slot.bridge.publication_profiling_enable()

    def draw(self):
        """Compatibility no-op; Draw Manager owns native PointCloud drawing."""

    def materialize_display_values(
            self, particles, mode, count, step=1, frame_id=None):
        cache_key = ((frame_id, mode, int(count), int(step))
                     if frame_id is not None else None)
        if (cache_key is not None
                and cache_key == getattr(self, "_materialization_cache_key", None)):
            return self._materialization_cache_values
        if self._display_kernels is None:
            self._display_kernels = {
                "HEIGHT": _make_height_sample_kernel(self.bridge.wp),
                "SPEED": _make_speed_sample_kernel(self.bridge.wp),
                "SCALAR": _make_scalar_sample_kernel(self.bridge.wp),
            }
        if self._display_values is None:
            self._display_values = self.bridge.wp.empty(
                self.capacity, dtype=float, device=self.bridge.device)
        if mode == "HEIGHT":
            source = particles.position
            kernel = self._display_kernels["HEIGHT"]
        elif mode == "SPEED":
            source = particles.velocity
            kernel = self._display_kernels["SPEED"]
        else:
            name = "time_offset" if mode == "TIME_OFFSET" else mode.lower()
            source = getattr(particles, name, None)
            if source is None:
                source = particles.attr(name)
            kernel = self._display_kernels["SCALAR"]
        self.bridge.wp.launch(
            kernel,
            dim=count,
            inputs=[source, self._display_values, int(step)],
            device=self.bridge.device,
            stream=self.bridge.stream,
        )
        values = self._display_values[:count].numpy()
        self.materialization_stats["reads"] += 1
        self.materialization_stats["bytes"] += count * 4
        self.materialization_stats["reason"] = f"particle_attribute:{mode}"
        self._materialization_cache_key = cache_key
        self._materialization_cache_values = values
        return values

    def close(self):
        errors = []
        failed_slots = []
        for slot in self._slots:
            try:
                slot.close()
            except Exception as error:
                errors.append(error)
                failed_slots.append(slot)
        self._slots = failed_slots
        if not failed_slots:
            self.pointcloud = None
            self._display_values = None
            self._materialization_cache_values = None
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                primary.add_note(str(error))
            raise primary


def replace_native_particle_presenter(
        previous,
        *,
        capacity,
        pointcloud,
        radius,
        positions,
        count=None,
        active_count=None,
        output_count=None,
        step=1):
    """Publish a replacement generation before retiring the previous one."""
    replacement = NativeParticlePresenter(capacity, pointcloud, radius)
    try:
        if active_count is None:
            replacement.publish(positions, count, step)
        else:
            replacement.publish_device_count(
                positions, active_count, output_count, step)
    except Exception as error:
        try:
            replacement.close()
        except Exception as cleanup_error:
            error.add_note(str(cleanup_error))
            _close_slots_with_retry(
                getattr(replacement, "_slots", ()), error)
        if previous is not None:
            try:
                previous._restore_active_binding()
            except Exception as restore_error:
                error.add_note(str(restore_error))
        raise
    if previous is not None:
        try:
            previous.close()
        except Exception as error:
            _close_slots_with_retry(
                getattr(previous, "_slots", ()), error)
            replacement.retirement_error = str(error)
    return replacement
