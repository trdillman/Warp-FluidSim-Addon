# Limit Speed [particle_speed_limit]
current = particle_velocity[particle_index]
speed = wp.length(current)
if speed > maximum_speed:
    particle_velocity[particle_index] = current * (maximum_speed / speed)
