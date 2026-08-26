# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/NVIDIA/warp/blob/v1.13.0/warp/examples/fem/example_apic_fluid.py
# Immutable revision: v1.13.0

@integrand
def integrate_fraction(s: Sample, phi: Field):
    return phi(s)

@integrand
def integrate_velocity(s: Sample, domain: Domain, u: Field,
                       velocities: wp.array[wp.vec3],
                       velocity_gradients: wp.array[wp.mat33], dt: float,
                       gravity: wp.vec3):
    """Transfer particle velocities to grid"""
    node_offset = domain(at_node(u, s)) - domain(s)
    vel_apic = velocities[s.qp_index] + velocity_gradients[s.qp_index] * node_offset
    vel_adv = vel_apic + dt * gravity
    return wp.dot(u(s), vel_adv)

@integrand
def update_particles(s: Sample, domain: Domain, grid_vel: Field, dt: float,
                     pos: wp.array[wp.vec3], pos_prev: wp.array[wp.vec3],
                     vel: wp.array[wp.vec3], vel_grad: wp.array[wp.mat33]):
    """Read particle velocity from grid and advect positions"""
    p_vel = grid_vel(s)
    vel_grad[s.qp_index] = grad(grid_vel, s)
    pos[s.qp_index] = pos_prev[s.qp_index] + dt * p_vel
    vel[s.qp_index] = p_vel
