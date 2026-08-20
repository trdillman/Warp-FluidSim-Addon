# Damp Velocity [velocity_damping]
factor = wp.clamp(1.0 - damping * solver_time_step, 0.0, 1.0)
particle_velocity[particle_index] = particle_velocity[particle_index] * factor
