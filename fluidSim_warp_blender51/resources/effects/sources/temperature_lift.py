# Apply Thermal Lift [temperature_lift]
direction_length = wp.length(direction)
if direction_length > 1.0e-6:
    direction_n = direction / direction_length
    excess = particle_temperature[particle_index] - ambient
    particle_velocity[particle_index] = particle_velocity[particle_index] + direction_n * (excess * buoyancy * solver_time_step)

# Cool Particle Temperature [temperature_cooling]
factor = wp.clamp(1.0 - cooling * solver_time_step, 0.0, 1.0)
particle_temperature[particle_index] = ambient + (particle_temperature[particle_index] - ambient) * factor
