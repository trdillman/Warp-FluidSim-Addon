# Apply Vortex [vortex_force]
offset = position - controller
axis_length = wp.length(axis)
distance = wp.length(offset)
if axis_length > 1.0e-6 and distance < radius:
    axis_n = axis / axis_length
    radial = offset - axis_n * wp.dot(offset, axis_n)
    radial_length = wp.length(radial)
    if radial_length > 1.0e-6:
        influence = wp.clamp(1.0 - radial_length / radius, 0.0, 1.0)
        tangent = wp.normalize(wp.cross(axis_n, radial))
        updated = particle_velocity[particle_index]
        updated = updated + tangent * (strength * influence * solver_time_step)
        updated = updated - radial * (suction * influence * solver_time_step)
        updated = updated + axis_n * (lift * influence * solver_time_step)
        particle_velocity[particle_index] = updated
