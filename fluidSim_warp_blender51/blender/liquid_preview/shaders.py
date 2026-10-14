"""Liquid Preview shaders, built only through ``GPUShaderCreateInfo``.

No ``bgl``, no ``#version`` pragmas: Blender generates the GLSL/SPIR-V
preamble for whichever backend (OpenGL, Vulkan) is running. Every shader reads
the shared ``LiquidPreviewParams`` uniform block; per-pass switches are push
constants. Depth is positive eye-space distance (0 means "no liquid").
"""

from __future__ import annotations

PARAMS_TYPEDEF = """
struct LiquidPreviewParams {
  mat4 view_mat;
  mat4 proj_mat;
  mat4 inv_proj;
  mat4 inv_view;
  vec4 target;
  vec4 sprite;
  vec4 range;
  vec4 absorb;
  vec4 tint;
  vec4 viewport;
};
"""
# target:   offscreen width, height, 1/width, 1/height
# sprite:   particle radius, perspective flag, tan(fov/2), whitewater radius
# range:    delta low, delta high, filter sigma (world), max filter radius px
# absorb:   Beer-Lambert coefficients rgb, refraction kappa
# tint:     absorption colour rgb, blue-noise frame
# viewport: destination x, y, width, height in pixels

_PROJECT_DEPTH = """
float window_depth(vec3 eye)
{
  vec4 clip = params.proj_mat * vec4(eye, 1.0);
  return clamp(clip.z / clip.w * 0.5 + 0.5, 0.0, 1.0);
}

float sprite_size(vec3 eye, float radius)
{
  float scale = params.proj_mat[1][1] * params.target.y;
  float dist = params.sprite.y > 0.5 ? max(-eye.z, 1e-4) : 1.0;
  return clamp(radius * scale / dist, 1.0, 512.0);
}
"""

SPRITE_VERT = _PROJECT_DEPTH + """
void main()
{
  vec4 eye = params.view_mat * vec4(pos, 1.0);
  eye_center = eye.xyz;
  gl_Position = params.proj_mat * eye;
  gl_PointSize = sprite_size(eye.xyz, params.sprite.x);
}
"""

DEPTH_FRAG = _PROJECT_DEPTH + """
void main()
{
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) {
    discard;
  }
  float r = params.sprite.x;
  vec3 eye = vec3(eye_center.xy + vec2(uv.x, -uv.y) * r,
                  eye_center.z + r * sqrt(1.0 - r2));
  gl_FragDepth = window_depth(eye);
  frag = vec4(-eye.z, 0.0, 0.0, 1.0);
}
"""

THICKNESS_FRAG = """
void main()
{
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) {
    discard;
  }
  ivec2 cell = ivec2(mod(gl_FragCoord.xy + params.tint.w * vec2(37.0, 17.0), 64.0));
  float jitter = texelFetch(noise_tex, cell, 0).r;
  float t = 2.0 * params.sprite.x * sqrt(1.0 - r2) * (0.75 + 0.5 * jitter);
  frag = vec4(t, 0.0, 0.0, t);
}
"""

FULLSCREEN_VERT = """
void main()
{
  gl_Position = vec4(co, 0.0, 1.0);
}
"""

_FILTER_COMMON = """
float filter_weight(float offset, float sigma_px)
{
  return exp(-offset * offset / (2.0 * sigma_px * sigma_px));
}

int filter_radius(float depth)
{
  float denom = 2.0 * params.sprite.z * (params.sprite.y > 0.5 ? depth : 1.0);
  float r = ceil(params.target.y * params.range.z / max(denom, 1e-6));
  return int(min(r, params.range.w));
}
"""

FILTER_FRAG = _FILTER_COMMON + """
/* Narrow-range filter (Truong & Yuksel 2018), one separable direction. */
void main()
{
  ivec2 size = textureSize(depth_tex, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float d = texelFetch(depth_tex, p, 0).r;
  if (d <= 0.0) {
    frag = vec4(0.0);
    return;
  }
  int radius = filter_radius(d);
  if (radius <= 0) {
    frag = vec4(d, 0.0, 0.0, 1.0);
    return;
  }
  float sigma_px = max(float(radius) * 0.5, 0.5);
  float lo = d - params.range.x;
  float hi = d + params.range.y;
  float sum = 0.0;
  float wsum = 0.0;
  ivec2 dir = ivec2(filter_dir);
  for (int i = -radius; i <= radius; i++) {
    ivec2 q = clamp(p + dir * i, ivec2(0), size - 1);
    float zi = texelFetch(depth_tex, q, 0).r;
    if (zi <= 0.0 || zi > hi) {
      continue;
    }
    float w = filter_weight(float(i), sigma_px);
    sum += w * max(zi, lo);
    wsum += w;
  }
  frag = vec4(sum / max(wsum, 1e-8), 0.0, 0.0, 1.0);
}
"""

CLEANUP_FRAG = _FILTER_COMMON + """
/* Narrow-range clean-up: a small 2D pass that removes separable artefacts. */
void main()
{
  ivec2 size = textureSize(depth_tex, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float d = texelFetch(depth_tex, p, 0).r;
  if (d <= 0.0) {
    frag = vec4(0.0);
    return;
  }
  int radius = min(filter_radius(d), 2);
  float sigma_px = max(float(radius) * 0.5, 0.5);
  float lo = d - params.range.x;
  float hi = d + params.range.y;
  float sum = 0.0;
  float wsum = 0.0;
  for (int y = -radius; y <= radius; y++) {
    for (int x = -radius; x <= radius; x++) {
      ivec2 q = clamp(p + ivec2(x, y), ivec2(0), size - 1);
      float zi = texelFetch(depth_tex, q, 0).r;
      if (zi <= 0.0 || zi > hi) {
        continue;
      }
      float w = filter_weight(length(vec2(x, y)), sigma_px);
      sum += w * max(zi, lo);
      wsum += w;
    }
  }
  frag = vec4(sum / max(wsum, 1e-8), 0.0, 0.0, 1.0);
}
"""

COMPOSITE_FRAG = _PROJECT_DEPTH + """
vec3 eye_position(ivec2 p, float depth)
{
  vec2 ndc = (vec2(p) + 0.5) * params.target.zw * 2.0 - 1.0;
  vec4 v = params.inv_proj * vec4(ndc, 0.0, 1.0);
  v /= v.w;
  if (params.sprite.y > 0.5) {
    return v.xyz * (depth / max(-v.z, 1e-6));
  }
  return vec3(v.xy, -depth);
}

/* Min-gradient silhouette rule: use the one-sided difference with the
   smaller depth jump, and never difference against empty texels. */
vec3 surface_derivative(ivec2 p, ivec2 step_dir, vec3 center, ivec2 size)
{
  ivec2 pa = clamp(p + step_dir, ivec2(0), size - 1);
  ivec2 pb = clamp(p - step_dir, ivec2(0), size - 1);
  float da = texelFetch(depth_tex, pa, 0).r;
  float db = texelFetch(depth_tex, pb, 0).r;
  vec3 forward = da > 0.0 ? eye_position(pa, da) - center : vec3(0.0);
  vec3 backward = db > 0.0 ? center - eye_position(pb, db) : vec3(0.0);
  if (da <= 0.0) {
    return backward;
  }
  if (db <= 0.0) {
    return forward;
  }
  return abs(forward.z) < abs(backward.z) ? forward : backward;
}

void main()
{
  ivec2 size = textureSize(depth_tex, 0);
  vec2 uv = (gl_FragCoord.xy - params.viewport.xy) / params.viewport.zw;
  ivec2 p = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
  float d = texelFetch(depth_tex, p, 0).r;
  if (d <= 0.0) {
    discard;
  }
  vec3 eye = eye_position(p, d);
  vec3 ddx = surface_derivative(p, ivec2(1, 0), eye, size);
  vec3 ddy = surface_derivative(p, ivec2(0, 1), eye, size);
  vec3 n = cross(ddx, ddy);
  vec3 view = params.sprite.y > 0.5 ? normalize(-eye) : vec3(0.0, 0.0, 1.0);
  n = dot(n, n) > 1e-20 ? normalize(n) : view;
  if (dot(n, view) < 0.0) {
    n = -n;
  }
  float ndv = clamp(dot(n, view), 0.0, 1.0);

  /* Thickness, read through a refraction offset along the normal. */
  float t0 = texelFetch(thickness_tex, p, 0).r;
  vec2 refracted = uv + n.xy * t0 * params.absorb.w;
  ivec2 pr = clamp(ivec2(refracted * vec2(size)), ivec2(0), size - 1);
  float t = texelFetch(thickness_tex, pr, 0).r;
  if (t <= 0.0) {
    t = t0;
  }
  vec3 transmit = exp(-params.absorb.rgb * t);
  float transmit_mean = (transmit.r + transmit.g + transmit.b) / 3.0;

  /* Schlick Fresnel, R0 for water. */
  float fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
  vec3 reflected = mat3(params.inv_view) * reflect(-view, n);
  float up = clamp(reflected.z * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = mix(vec3(0.32, 0.34, 0.38), vec3(0.85, 0.92, 1.0), up);
  vec3 light = normalize(view + vec3(0.35, 0.55, 0.0));
  float spec = pow(clamp(dot(n, normalize(light + view)), 0.0, 1.0), 80.0) * 0.6;
  vec3 body = params.tint.rgb * (0.55 + 0.45 * ndv);

  float alpha = 1.0 - (1.0 - fresnel) * transmit_mean;
  vec3 color = fresnel * sky + (1.0 - fresnel) * (1.0 - transmit_mean) * body + spec;
  gl_FragDepth = window_depth(eye);
  frag = vec4(color, clamp(alpha + spec, 0.0, 1.0));
}
"""

WHITEWATER_VERT = """
void main()
{
  vec4 eye = params.view_mat * vec4(pos, 1.0);
  eye_center = eye.xyz;
  ww_type = wtype;
  gl_Position = params.proj_mat * eye;
  float size_scale = (wtype > 1.5 && wtype < 2.5) ? 0.6 : 1.0;
  float scale = params.proj_mat[1][1] * params.viewport.w;
  float dist = params.sprite.y > 0.5 ? max(-eye.z, 1e-4) : 1.0;
  gl_PointSize = clamp(params.sprite.w * size_scale * scale / dist, 1.0, 64.0);
}
"""

WHITEWATER_FRAG = """
void main()
{
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(uv, uv);
  if (r2 > 1.0) {
    discard;
  }
  vec3 color = vec3(1.0);
  float alpha = 0.85;
  if (ww_type > 0.5 && ww_type < 1.5) {
    color = mix(params.tint.rgb, vec3(1.0), 0.35);
    alpha = 0.55;
  }
  else if (ww_type > 1.5 && ww_type < 2.5) {
    alpha = 1.0;
  }
  ivec2 size = textureSize(depth_tex, 0);
  vec2 screen = (gl_FragCoord.xy - params.viewport.xy) / params.viewport.zw;
  ivec2 p = clamp(ivec2(screen * vec2(size)), ivec2(0), size - 1);
  float liquid = texelFetch(depth_tex, p, 0).r;
  if (liquid > 0.0 && -eye_center.z > liquid + 2.0 * params.sprite.x) {
    alpha *= 0.35; /* seen through the liquid body */
  }
  alpha *= 1.0 - r2 * r2;
  frag = vec4(color * alpha, alpha);
}
"""


def _info(gpu, name, vertex_source, fragment_source, *, vertex_inputs,
          samplers=(), push_constants=(), interface=None, depth_write=False):
    info = gpu.types.GPUShaderCreateInfo()
    info.typedef_source(PARAMS_TYPEDEF)
    info.uniform_buf(0, "LiquidPreviewParams", "params")
    for slot, (kind, attr) in enumerate(vertex_inputs):
        info.vertex_in(slot, kind, attr)
    for slot, sampler in enumerate(samplers):
        info.sampler(slot, "FLOAT_2D", sampler)
    for kind, constant in push_constants:
        info.push_constant(kind, constant)
    if interface is not None:
        stage = gpu.types.GPUStageInterfaceInfo(f"liquid_preview_{name}_iface")
        for qualifier, kind, varying in interface:
            getattr(stage, qualifier)(kind, varying)
        info.vertex_out(stage)
    info.fragment_out(0, "VEC4", "frag")
    if depth_write:
        info.depth_write("ANY")
    info.vertex_source(vertex_source)
    info.fragment_source(fragment_source)
    return info


def build_shaders(gpu):
    """Compile every Liquid Preview shader; raises on any backend failure."""
    sprite_iface = (("flat", "VEC3", "eye_center"),)
    infos = {
        "depth": _info(
            gpu, "depth", SPRITE_VERT, DEPTH_FRAG,
            vertex_inputs=(("VEC3", "pos"),), interface=sprite_iface,
            depth_write=True),
        "thickness": _info(
            gpu, "thickness", SPRITE_VERT, THICKNESS_FRAG,
            vertex_inputs=(("VEC3", "pos"),), interface=sprite_iface,
            samplers=("noise_tex",)),
        "filter": _info(
            gpu, "filter", FULLSCREEN_VERT, FILTER_FRAG,
            vertex_inputs=(("VEC2", "co"),), samplers=("depth_tex",),
            push_constants=(("VEC2", "filter_dir"),)),
        "cleanup": _info(
            gpu, "cleanup", FULLSCREEN_VERT, CLEANUP_FRAG,
            vertex_inputs=(("VEC2", "co"),), samplers=("depth_tex",)),
        "composite": _info(
            gpu, "composite", FULLSCREEN_VERT, COMPOSITE_FRAG,
            vertex_inputs=(("VEC2", "co"),),
            samplers=("depth_tex", "thickness_tex"), depth_write=True),
        "whitewater": _info(
            gpu, "whitewater", WHITEWATER_VERT, WHITEWATER_FRAG,
            vertex_inputs=(("VEC3", "pos"), ("FLOAT", "wtype")),
            samplers=("depth_tex",),
            interface=(("flat", "VEC3", "eye_center"),
                       ("flat", "FLOAT", "ww_type"))),
    }
    return {name: gpu.shader.create_from_info(info)
            for name, info in infos.items()}
