"""CUDA reference path integrator for the P1 feasibility subset.

Software BVH traversal through Warp; no OptiX/RTX claim. BSDF-only sampling
and a hard path-length limit are deliberate P1 limits. MIS, explicit light
sampling, rough transmission and nested media are not implemented here.
"""
import warp as wp


@wp.func
def unproject(matrix: wp.mat44, x: float, y: float, z: float):
    point = matrix * wp.vec4(x, y, z, 1.0)
    return wp.vec3(point[0], point[1], point[2]) / point[3]


@wp.func
def dielectric_fresnel(cos_i: float, eta_i: float, eta_t: float):
    ratio = eta_i / eta_t
    sin_t2 = ratio * ratio * wp.max(0.0, 1.0 - cos_i * cos_i)
    reflectance = float(1.0)
    if sin_t2 < 1.0:
        cos_t = wp.sqrt(wp.max(0.0, 1.0 - sin_t2))
        rs = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t)
        rp = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t)
        reflectance = 0.5 * (rs * rs + rp * rp)
    return reflectance


@wp.kernel
def trace_paths(
    mesh: wp.uint64, triangle_count: int,
    material_indices: wp.array(dtype=int), kinds: wp.array(dtype=int),
    colors: wp.array(dtype=wp.vec3), emissions: wp.array(dtype=wp.vec3),
    iors: wp.array(dtype=float), world: wp.vec3, inverse_vp: wp.mat44,
    width: int, height: int, first_sample: int, sample_count: int,
    max_bounces: int, seed: int,
    accumulation: wp.array(dtype=wp.vec3), rgba: wp.array(dtype=wp.vec4),
):
    pixel = wp.tid()
    x = pixel % width
    y = pixel // width
    total = accumulation[pixel]
    for sample in range(sample_count):
        rng = wp.rand_init(seed + first_sample + sample, pixel)
        sx = 2.0 * (float(x) + wp.randf(rng)) / float(width) - 1.0
        sy = 2.0 * (float(y) + wp.randf(rng)) / float(height) - 1.0
        origin = unproject(inverse_vp, sx, sy, -1.0)
        far = unproject(inverse_vp, sx, sy, 1.0)
        direction = wp.normalize(far - origin)
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)
        for bounce in range(max_bounces):
            found = bool(False)
            distance = float(0.0)
            normal = wp.vec3(0.0, 0.0, 1.0)
            face = int(0)
            if triangle_count > 0:
                hit = wp.mesh_query_ray(mesh, origin, direction, 1.0e20)
                found = hit.result
                distance = hit.t
                normal = hit.normal
                face = hit.face
            if not found:
                radiance = radiance + wp.cw_mul(throughput, world)
                break
            p = origin + distance * direction
            outward = wp.normalize(normal)
            entering = wp.dot(direction, outward) < 0.0
            n = outward
            if not entering:
                n = -outward
            material = material_indices[face]
            radiance = radiance + wp.cw_mul(throughput, emissions[material])
            if kinds[material] == 2:
                break
            color = colors[material]
            if kinds[material] == 1:
                eta_i = float(1.0)
                eta_t = iors[material]
                if not entering:
                    eta_i = iors[material]
                    eta_t = 1.0
                cos_i = wp.clamp(-wp.dot(direction, n), 0.0, 1.0)
                ratio = eta_i / eta_t
                sin_t2 = ratio * ratio * wp.max(0.0, 1.0 - cos_i * cos_i)
                fresnel = dielectric_fresnel(cos_i, eta_i, eta_t)
                if sin_t2 >= 1.0 or wp.randf(rng) < fresnel:
                    direction = wp.normalize(direction + 2.0 * cos_i * n)
                else:
                    direction = wp.normalize(ratio * direction +
                        (ratio * cos_i - wp.sqrt(wp.max(0.0, 1.0 - sin_t2))) * n)
                    # Radiance transport across the interface. The inverse
                    # factor cancels on exit for an isolated closed dielectric.
                    throughput = throughput * (ratio * ratio)
                throughput = wp.cw_mul(throughput, color)
            else:
                radius = wp.sqrt(wp.randf(rng))
                phi = 6.283185307179586 * wp.randf(rng)
                axis = wp.vec3(0.0, 0.0, 1.0)
                if wp.abs(n[2]) > 0.999:
                    axis = wp.vec3(0.0, 1.0, 0.0)
                tangent = wp.normalize(wp.cross(axis, n))
                bitangent = wp.cross(n, tangent)
                direction = wp.normalize(tangent * (radius * wp.cos(phi)) +
                    bitangent * (radius * wp.sin(phi)) + n * wp.sqrt(wp.max(0.0, 1.0 - radius * radius)))
                throughput = wp.cw_mul(throughput, color)
            epsilon = wp.max(1.0e-5, 1.0e-6 * wp.length(p))
            offset = n * epsilon
            if wp.dot(direction, n) < 0.0:
                offset = -offset
            origin = p + offset
            if bounce >= 3:
                survival = wp.clamp(wp.max(throughput[0], wp.max(throughput[1], throughput[2])), 0.05, 0.95)
                if wp.randf(rng) > survival:
                    break
                throughput = throughput / survival
        total = total + radiance
    accumulation[pixel] = total
    color = total / float(first_sample + sample_count)
    rgba[pixel] = wp.vec4(color[0], color[1], color[2], 1.0)
