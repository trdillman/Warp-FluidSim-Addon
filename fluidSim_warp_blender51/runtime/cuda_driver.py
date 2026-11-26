"""Checked public CUDA driver operations on Warp-owned contexts and event handles."""
import ctypes
import sys
from functools import lru_cache


def check_cuda(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA result {status}")


class CudaDriver:
    def __init__(self):
        self.library = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
                        else ctypes.CDLL("libcuda.so.1"))
        ptr, uint = ctypes.c_void_p, ctypes.c_uint
        for name, args in {
            "cuStreamCreate": (ctypes.POINTER(ptr), uint),
            "cuStreamDestroy_v2": (ptr,),
            "cuEventRecord": (ptr, ptr),
            "cuStreamWaitEvent": (ptr, ptr, uint),
            "cuEventQuery": (ptr,),
        }.items():
            fn = getattr(self.library, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int

    def create_nonblocking(self, wp, device):
        handle = ctypes.c_void_p()
        with wp.ScopedDevice(device):
            check_cuda(self.library.cuStreamCreate(ctypes.byref(handle), 1), "cuStreamCreate")
        if not handle.value:
            raise RuntimeError("cuStreamCreate returned a null stream")
        return handle.value

    def destroy(self, wp, device, handle):
        with wp.ScopedDevice(device):
            return self.library.cuStreamDestroy_v2(handle)

    def record(self, wp, device, stream, event):
        with wp.ScopedDevice(device):
            check_cuda(self.library.cuEventRecord(event.cuda_event, stream.cuda_stream), "cuEventRecord")

    def wait(self, wp, device, stream, event):
        with wp.ScopedDevice(device):
            check_cuda(self.library.cuStreamWaitEvent(stream.cuda_stream, event.cuda_event, 0),
                       "cuStreamWaitEvent")

    def completed(self, wp, device, event):
        with wp.ScopedDevice(device):
            result = self.library.cuEventQuery(event.cuda_event)
        if result == 600:  # CUDA_ERROR_NOT_READY
            return False
        check_cuda(result, "cuEventQuery")
        return True


@lru_cache(maxsize=1)
def cuda_driver():
    return CudaDriver()
