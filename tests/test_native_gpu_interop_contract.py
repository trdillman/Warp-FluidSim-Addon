import ctypes
import inspect

from fluidSim_warp_blender51.blender import native_interop


def test_cuda_driver_abi_layout_matches_cuda_12_9_x64():
    assert ctypes.sizeof(native_interop._ExternalMemoryHandleDesc) == 104
    assert ctypes.sizeof(native_interop._ExternalMemoryBufferDesc) == 88
    assert ctypes.sizeof(native_interop._ExternalSemaphoreHandleDesc) == 96
    assert ctypes.sizeof(native_interop._ExternalSemaphoreSignalParams) == 144
    assert ctypes.sizeof(native_interop._ExternalSemaphoreWaitParams) == 144


def test_cuda_uuid_parser_matches_blender_bytes():
    value = "GPU-5d09f77b-a936-a6d9-054d-0324aef67adb"
    assert native_interop._uuid_bytes(value) == bytes.fromhex(
        "5d09f77ba936a6d9054d0324aef67adb")


def test_native_publish_uses_device_copy_not_numpy_materialization():
    source = inspect.getsource(native_interop.CudaExternalBuffer.publish)
    assert ".copy(" in source
    assert ".numpy(" not in source


def test_zero_particle_publish_does_not_start_a_handoff():
    presenter = native_interop.NativeParticlePresenter.__new__(
        native_interop.NativeParticlePresenter)
    presenter.active_count = 12
    presenter.pending_acquire = False
    presenter.value = 4
    assert presenter.publish(object(), 0) is True
    assert presenter.active_count == 0
    assert presenter.value == 4
    assert presenter.pending_acquire is False
