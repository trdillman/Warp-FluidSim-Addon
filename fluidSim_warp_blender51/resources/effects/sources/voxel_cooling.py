# Cool Active Voxels [voxel_cooling]
factor = wp.clamp(1.0 - cooling * real_time_step, 0.0, 1.0)
temperature[flat_voxel_index] = ambient + (temperature[flat_voxel_index] - ambient) * factor
