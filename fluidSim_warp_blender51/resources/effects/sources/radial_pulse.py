# Apply Pulse [radial_pulse]
offset = position - controller
distance = wp.length(offset)
if distance > 1.0e-6 and distance < radius:
    influence = wp.clamp(1.0 - distance / radius, 0.0, 1.0)
    direction = offset / distance
    particle_velocity[particle_index] = particle_velocity[particle_index] + direction * (strength * influence * solver_time_step)
