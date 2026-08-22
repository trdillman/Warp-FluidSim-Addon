# Apply Split [planar_split]
normal_length = wp.length(normal)
if normal_length > 1.0e-6:
    normal_n = normal / normal_length
    signed_distance = wp.dot(position - controller, normal_n)
    absolute_distance = wp.abs(signed_distance)
    if absolute_distance < width:
        side = 1.0
        if signed_distance < 0.0:
            side = -1.0
        influence = wp.clamp(1.0 - absolute_distance / width, 0.0, 1.0)
        particle_velocity[particle_index] = particle_velocity[particle_index] + normal_n * (side * strength * influence * solver_time_step)
