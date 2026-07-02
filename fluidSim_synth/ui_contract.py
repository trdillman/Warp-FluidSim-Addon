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
