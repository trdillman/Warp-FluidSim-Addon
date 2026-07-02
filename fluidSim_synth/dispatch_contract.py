"""Runtime dispatch contract for the Warp port.

This is a synthesized replacement for the original OpenCL command-queue path.
"""

ORIGINAL_DISPATCH_MODEL = {
    "program_build": "Compile a shared kernel source for a selected compute device.",
    "kernel_lookup": "Resolve kernels by string identity.",
    "argument_binding": "Bind device buffers, constant buffers, and scalar values before launch.",
    "ndrange_launch": "Launch one-, two-, or three-dimensional work domains.",
    "host_device_sync": "Write inputs, read outputs, and synchronize command queues where requested.",
    "queue_lifetime": "Create, reuse, flush, and finish a compute command queue.",
}

WARP_DISPATCH_MODEL = {
    "program_build": "JIT-compile @wp.kernel functions on first use for the active CUDA device.",
    "kernel_lookup": "Register Python kernel callables by FluidSim kernel identity.",
    "argument_binding": "Use typed wp.array buffers, scalar parameters, and small context structs/dataclasses.",
    "ndrange_launch": "Use wp.launch for flat domains and tiled kernels for workgroup-style reductions/shared memory.",
    "host_device_sync": "Keep simulation buffers device-resident; read back only diagnostics, UI summaries, and render/mesh outputs.",
    "queue_lifetime": "Use Warp streams/events where needed to model compute, copy, and profiling tracks.",
}

REQUIRED_GPU_POLICY = {
    "backend": "NVIDIA Warp/CUDA",
    "simulation_buffers": "GPU resident",
    "cpu_allowed_for": ["Blender API", "orchestration", "validation", "profiling", "minimal data transfer"],
    "cache_workflow": "optional export/performance tool only; not the core playback model",
}
