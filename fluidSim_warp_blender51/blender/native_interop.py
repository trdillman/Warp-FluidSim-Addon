"""Vulkan/CUDA particle presentation without CPU position materialization."""

from __future__ import annotations

import ctypes


class NativeInteropUnavailable(RuntimeError):
    """Raised when the custom Blender Vulkan interop surface is unavailable."""


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
        self.check(self.map_memory(ctypes.byref(mapped), memory,
                                   ctypes.byref(buffer_desc)),
                   "cuExternalMemoryGetMappedBuffer")
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
        self.closed = False
        self.stats = {
            "imports": 0,
            "maps": 0,
            "published_frames": 0,
            "published_bytes": 0,
            "bulk_host_position_bytes": 0,
        }

        self.driver.push(self.context)
        try:
            self.memory, self.mapped_pointer = self.driver.import_external_memory(
                descriptor["memory_handle"], descriptor["memory_size"])
            self.release = self.driver.import_timeline(
                descriptor["release_semaphore_handle"])
            self.completion = self.driver.import_timeline(
                descriptor["completion_semaphore_handle"])
        finally:
            self.driver.pop()
        self.positions = wp.array(
            ptr=self.mapped_pointer,
            capacity=descriptor["memory_size"],
            shape=(capacity,),
            dtype=wp.vec3,
            device=self.device,
        )
        self.stats["imports"] = 3
        self.stats["maps"] = 1

    def publish(self, source, count, value):
        self.driver.push(self.context)
        try:
            self.driver.wait(self.release, value, self.stream.cuda_stream)
            self.wp.copy(self.positions, source,
                         count=count, stream=self.stream)
            self.driver.signal(self.completion, value, self.stream.cuda_stream)
        finally:
            self.driver.pop()
        self.stats["published_frames"] += 1
        self.stats["published_bytes"] += count * 12

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.wp.synchronize_stream(self.stream)
        self.driver.push(self.context)
        try:
            for semaphore in (self.completion, self.release):
                if semaphore.value:
                    self.driver.check(
                        self.driver.destroy_semaphore(semaphore),
                        "cuDestroyExternalSemaphore",
                    )
            if self.mapped_pointer:
                self.driver.check(self.driver.free_memory(self.mapped_pointer),
                                  "cuMemFree")
            if self.memory.value:
                self.driver.check(self.driver.destroy_memory(self.memory),
                                  "cuDestroyExternalMemory")
        finally:
            self.driver.pop()


class NativeParticlePresenter:
    """Publish Warp particles into a Blender-owned VBO and draw it directly."""

    def __init__(self, capacity, point_size=3.0):
        import bpy
        import gpu

        if not hasattr(gpu.types.GPUVertBuf, "external_descriptor_get"):
            raise NativeInteropUnavailable(
                "Blender does not expose the native GPU interop API")
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type="F32", len=3, fetch_mode="FLOAT")
        self.vbo = gpu.types.GPUVertBuf(
            format=fmt, len=capacity, external=True)
        self.batch = gpu.types.GPUBatch(type="POINTS", buf=self.vbo)
        self.shader = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        self.bridge = CudaExternalBuffer(
            self.vbo.external_descriptor_get(), capacity)
        self.point_size = float(point_size)
        self.active_count = 0
        self.value = 0
        self.pending_acquire = False
        self.handle = bpy.types.SpaceView3D.draw_handler_add(
            self.draw, (), "WINDOW", "POST_VIEW")

    def publish(self, positions, count):
        if count <= 0:
            self.active_count = 0
            return True
        if self.pending_acquire:
            return False
        self.value += 1
        self.vbo.external_release(self.value)
        self.bridge.publish(positions, count, self.value)
        self.active_count = int(count)
        self.pending_acquire = True
        return True

    def draw(self):
        if self.active_count <= 0:
            return
        import gpu

        if self.pending_acquire:
            self.vbo.external_acquire(self.value)
        self.shader.bind()
        self.shader.uniform_float("color", (0.08, 0.45, 1.0, 1.0))
        self.shader.uniform_float("size", self.point_size)
        self.batch.draw_range(
            self.shader, elem_start=0, elem_count=self.active_count)
        if self.pending_acquire:
            self.vbo.external_submit(self.value)
            self.pending_acquire = False

    def close(self):
        import bpy

        if self.handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            self.handle = None
        self.bridge.close()
        self.batch = None
        self.vbo = None
