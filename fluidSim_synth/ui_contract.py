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
    "Sources",
    "Colliders / Sinks",
    "Forces / Attraction",
    "Pressure Projection",
    "Viscosity",
    "Surface / Whitewater",
    "Realtime Surfacing / Particles",
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

REALTIME_SURFACING_PARTICLES_SECTION = {
    "surface_stack": [
        "particle_radius_scale",
        "surface_dilate_iterations",
        "surface_dilate_strength",
        "surface_erode_iterations",
        "surface_erode_strength",
        "surface_smooth_type",
        "surface_smooth_iterations",
        "surface_smooth_strength",
        "surface_attribute_mode",
        "surface_material",
    ],
    "particles": [
        "show_particles",
        "particle_display_fraction",
        "particle_display_scale",
        "particle_attribute_mode",
        "particle_material",
    ],
    "whitewater_display": [
        "show_whitewater",
        "whitewater_enabled",
        "whitewater_display_scale",
        "whitewater_attribute_mode",
        "whitewater_material",
    ],
    "legacy_hidden_compatibility": [
        "surface_filter_type",
        "surface_filter_iterations",
        "surface_filter_strength",
    ],
}
