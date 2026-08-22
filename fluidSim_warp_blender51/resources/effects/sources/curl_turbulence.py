# Apply Turbulence [curl_turbulence]
sample = position * frequency + wp.vec3(seed, seed * 1.37, seed * 2.11)
phase = time * drift
noise = wp.vec3(
    wp.sin(sample[1] + phase) - wp.sin(sample[2] - phase),
    wp.sin(sample[2] + phase * 1.17) - wp.sin(sample[0] - phase),
    wp.sin(sample[0] + phase * 0.73) - wp.sin(sample[1] - phase * 1.31),
)
particle_velocity[particle_index] = particle_velocity[particle_index] + noise * (strength * solver_time_step)
