"""Profiling and benchmark schema synthesized from the original tools."""

PROFILE_COMPUTE_FIELDS = [
    "deviceName",
    "api",
    "usedHostBufferTotalMemoryTally",
    "usedBufferTotalMemoryTally",
    "usedBufferDeviceMemoryTally",
    "usedBufferSharedMemoryTally",
    "deviceHasUnifiedMemory",
    "deviceUsedSharedMemory",
]

PROFILE_FRAME_FIELDS = [
    "stepTime",
    "activeVoxels",
    "initializedVoxels",
    "totalVoxels",
    "activeParticles",
    "totalParticles",
    "voxelsPerSecond",
    "voxelsPerFrame",
    "particlesPerSecond",
    "particlesPerFrame",
    "usedParticleAllocation",
    "usedVoxelAllocation",
    "activeInitializedVoxelRatio",
    "particleInitializedVoxelRatio",
    "particleVoxelFillRatio",
    "timers",
]

TIMER_TRACK_ORDER = [
    "dcc", "dcc_total", "dcc_other", "input", "solver", "script", "compute", "blit",
    *[f"compute_{i}" for i in range(16)],
    *[f"blit_{i}" for i in range(16)],
    *[f"eventWait_{i}" for i in range(16)],
    "queue_commit", "queue_finish", "queue_compute", "queue_blit", "queue", "output",
    *[f"debug_{i}" for i in range(16)],
    "total",
]

BENCHMARK_FIELDS = [
    "totalTime",
    "timeStepAverage",
    "timeStepMinimum",
    "timeStepMaximum",
    "voxelsPerSecondAverage",
    "voxelsPerSecondMinimum",
    "voxelsPerSecondMaximum",
    "activeVoxelsAverage",
    "activeVoxelsMinimum",
    "activeVoxelsMaximum",
    "result",
]

BENCHMARK_COMPARISON_RULES = {
    "primary_exact_match_fields": ["activeVoxelsAverage", "activeVoxelsMinimum", "activeVoxelsMaximum"],
    "performance_fields": ["timeStepAverage", "voxelsPerSecondAverage"],
    "recommended_blender_output": "JSON plus viewport diagnostics overlay",
}
