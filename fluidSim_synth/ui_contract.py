"""Blender 5.1 UI/property behavior synthesized from the original Houdini wrapper."""

PARTICLE_ATTRIBUTE_RAMP_SOURCES = [
    ("temperature", "Temperature"),
    ("viscosity", "Viscosity"),
    ("age", "Age"),
    ("life", "Life"),
]

FIELD_RAMP_SOURCES = [("pressure", "Pressure")]

EXTRA_PARTICLE_ATTRIBUTE_RULE = {
    "include_when_components": 1,
    "include_when_type_in": ["Float", "Half"],
    "label_source": "attribute name",
}

BLENDER_UI_PANELS = [
    "Simulation",
    "Domain / Sparse Grid",
    "Particles",
    "Liquid Emitters",
    "Liquid Sinks",
    "Colliders",
    "Forces",
    "Domain Height Fill",
    "Point / Particle Injection",
    "Advanced Field Sources",
    "Pressure Projection",
    "Viscosity",
    "Whitewater",
    "Realtime Surfacing / Particles",
    "Realtime Surfacing / Particles > Advanced Surface",
    "Viewport / Render Output",
    "Custom Warp Snippets",
    "Diagnostics / Profiling",
    "Validation / Benchmarks",
]

OMIT_OR_REPLACE_FROM_ORIGINAL = [
    "Houdini hou module usage",
    "HScript command wrappers",
    "Houdini PythonPanel profiler UI",
    "Houdini node-type compatibility checks",
    "Houdini license login/logout/license-file UI",
]

BLENDER_DATA_INTEGRATIONS = {
    "sources": ["mesh", "volume", "point cloud", "particle system", "curve/geometry nodes output where accessible"],
    "colliders": ["mesh SDF", "signed distance volume", "animated transforms", "velocity transfer"],
    "outputs": ["liquid surface mesh or volume", "whitewater particles/points", "debug fields", "profiling overlay"],
    "render": ["material assignment", "viewport visibility", "render visibility", "optional cache/export"],
}

SURFACE_PANEL = {
    "main": [
        "surface_enabled",
        "surface_resolution_mode",
        "surface_resolution_multiplier",
        "surface_voxel_size",
        "particle_radius_scale",
        "surface_splat_radius",
        "surface_filter_type",
        "surface_filter_iterations",
        "surface_filter_strength",
        "surface_dilate_iterations",
        "surface_dilate_strength",
        "surface_erode_iterations",
        "surface_erode_strength",
        "surface_smooth_type",
        "surface_smooth_iterations",
        "surface_smooth_strength",
        "surface_update_cadence",
        "live_surface_publish_cadence",
        "show_surface",
        "surface_attribute_mode",
        "surface_material",
    ],
    "advanced": [
        "surface_leaf_capacity_scale",
        "surface_max_dense_dim",
        "surface_build_lowres_sdf",
    ],
}

LIQUID_PARTICLES_PANEL = [
    "show_particles",
    "particle_display_fraction",
    "particle_display_scale",
    "particle_attribute_mode",
    "particle_material",
]

WHITEWATER_PANEL = {
    "simulation": [
        "whitewater_enabled",
        "max_whitewater",
        "ww_churn_gain",
        "ww_churn_decay",
        "ww_churn_threshold",
        "ww_emission_rate",
        "ww_min_speed",
        "ww_life_min",
        "ww_life_max",
        "ww_bubble_drag",
        "ww_spray_drag",
    ],
    "display": [
        "show_whitewater",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    ],
}
