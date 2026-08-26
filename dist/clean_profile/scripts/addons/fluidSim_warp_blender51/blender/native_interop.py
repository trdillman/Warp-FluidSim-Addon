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
        else:
            target[index] = wp.vec4(0.0, 0.0, 0.0, 0.0)

    _publish.__annotations__ = {
        "source": wp.array(dtype=wp.vec3),
        "active_count": wp.array(dtype=wp.int32),
        "target": wp.array(dtype=wp.vec4),
        "radius": float,
        "step": int,
    }
    return wp.kernel(_publish)


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


class CudaExternalBuffer:
    """One persistently imported Blender VBO and two directional timelines."""

    def __init__(self, descriptor, capacity):
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
            self.publish_kernel = _make_position_radius_kernel(wp)
            self.counted_publish_kernel = _make_counted_position_radius_kernel(wp)
            self.clear_kernel = _make_clear_position_radius_kernel(wp)
            self.visible_count = None
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(str(cleanup_error))
            raise
        self.stats["imports"] = 3
        self.stats["maps"] = 1

    def publish(self, source, count, value, radius, step=1):
        clear_end = self.capacity if self.visible_count is None else self.visible_count
        clear_count = max(0, clear_end - count)
        self.driver.push(self.context)
        self.context_pushed = True
        try:
            self.driver.wait(self.release, value, self.stream.cuda_stream)
            if count > 0:
                self.wp.launch(
                    self.publish_kernel,
                    dim=count,
                    inputs=[source, self.positions, float(radius), int(step)],
                    device=self.device,
                    stream=self.stream,
                )
            if clear_count > 0:
                self.wp.launch(
                    self.clear_kernel,
                    dim=clear_count,
                    inputs=[self.positions, int(count)],
                    device=self.device,
                    stream=self.stream,
                )
            self.driver.signal(self.completion, value, self.stream.cuda_stream)
        finally:
            self.driver.pop()
            self.context_pushed = False
        self.visible_count = int(count)
        self.stats["published_frames"] += 1
        self.stats["published_bytes"] += (count + clear_count) * 16

    def publish_device_count(
            self, source, active_count, output_count, value, radius, step=1):
        self.driver.push(self.context)
        self.context_pushed = True
        try:
            self.driver.wait(self.release, value, self.stream.cuda_stream)
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
            self.driver.signal(self.completion, value, self.stream.cuda_stream)
        finally:
            self.driver.pop()
            self.context_pushed = False
        self.visible_count = int(output_count)
        self.stats["published_frames"] += 1
        self.stats["published_bytes"] += output_count * 16

    def close(self):
        if self.closed:
            return
        errors = []
        try:
            self.wp.synchronize_stream(self.stream)
        except Exception as error:
            errors.append(error)

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
        self.bridge = CudaExternalBuffer(
            self.vbo.external_descriptor_get(), capacity)
        self.pointcloud = None
        self.active_count = 0
        self.value = 0

    def bind_pointcloud(self, pointcloud):
        self.vbo._pointcloud_position_bind(pointcloud)
        self.pointcloud = pointcloud

    def publish(self, positions, count, radius, step=1):
        bytes_before = self.bridge.stats["published_bytes"]
        self.value += 1
        self.vbo.external_release(self.value)
        self.bridge.publish(positions, count, self.value, radius, step)
        self.vbo.external_acquire(self.value)
        self.vbo._pointcloud_position_count_set(count)
        self.vbo.external_submit(self.value)
        self.active_count = int(count)
        return self.bridge.stats["published_bytes"] - bytes_before

    def publish_device_count(
            self, positions, active_count, output_count, radius, step=1):
        bytes_before = self.bridge.stats["published_bytes"]
        self.value += 1
        self.vbo.external_release(self.value)
        self.bridge.publish_device_count(
            positions, active_count, output_count, self.value, radius, step)
        self.vbo.external_acquire(self.value)
        self.vbo._pointcloud_position_count_set(output_count)
        self.vbo.external_submit(self.value)
        self.active_count = int(output_count)
        return self.bridge.stats["published_bytes"] - bytes_before

    def close(self):
        errors = []
        if self.vbo is not None:
            try:
                self.vbo._pointcloud_position_unbind()
            except Exception as error:
                errors.append(error)
        try:
            self.bridge.close()
        except Exception as error:
            errors.append(error)
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

    def __init__(self, capacity, pointcloud, radius):
        import gpu

        _retry_deferred_slot_cleanup()
        if not hasattr(gpu.types.GPUVertBuf, "_pointcloud_position_bind"):
            raise NativeInteropUnavailable(
                "Blender does not expose native PointCloud interop")
        self._slots = []
        try:
            for _index in range(2):
                self._slots.append(_NativeParticleSlot(capacity))
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

    def publish_device_count(self, positions, active_count, output_count, step=1):
        next_slot_index = (self._slot_index + 1) % len(self._slots)
        previous_slot = self._slots[self._slot_index] if self._slot_index >= 0 else None
        slot = self._slots[next_slot_index]
        try:
            slot.bind_pointcloud(self.pointcloud)
            published_bytes = slot.publish_device_count(
                positions, active_count, output_count, self.radius, step)
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
    except Exception:
        replacement.close()
        raise
    if previous is not None:
        previous.close()
    return replacement
