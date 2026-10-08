"""Strict, versioned, allow-listed FluidSim preset schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

PRESET_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True)
class SettingSpec:
    value_type: type
    minimum: float | int | None = None
    maximum: float | int | None = None
    choices: tuple[str, ...] = ()

    def validate(self, key: str, value: Any) -> Any:
        if self.value_type is bool:
            if type(value) is not bool:
                raise ValueError(f"preset setting {key!r} must be a boolean")
            return value
        if self.value_type is int:
            if type(value) is not int:
                raise ValueError(f"preset setting {key!r} must be an integer")
            result: Any = int(value)
        elif self.value_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"preset setting {key!r} must be numeric")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"preset setting {key!r} must be finite")
        elif self.value_type is str:
            if not isinstance(value, str):
                raise ValueError(f"preset setting {key!r} must be a string")
            result = value
        else:  # defensive: the schema is audited and static
            raise RuntimeError(f"unsupported SettingSpec type for {key}")
        if self.minimum is not None and result < self.minimum:
            raise ValueError(f"preset setting {key!r} must be >= {self.minimum}")
        if self.maximum is not None and result > self.maximum:
            raise ValueError(f"preset setting {key!r} must be <= {self.maximum}")
        if self.choices and result not in self.choices:
            raise ValueError(
                f"preset setting {key!r} must be one of {', '.join(self.choices)}"
            )
        return result


# This is intentionally not an arbitrary RNA path map. New product settings
# require a reviewed schema addition and tests before a preset can assign them.
SETTING_SPECS: dict[str, SettingSpec] = {
    "voxel_size": SettingSpec(float, 0.005, 10.0),
    "leaf_capacity": SettingSpec(int, 64, 262_144),
    "topology_expand": SettingSpec(int, 0, 4),
    "max_particles": SettingSpec(int, 1_024, 50_000_000),
    "auto_grow_particles": SettingSpec(bool),
    "particles_per_voxel": SettingSpec(int, 0, 64),
    "sample_type": SettingSpec(str, choices=("PIC", "FLIP")),
    "flip_ratio": SettingSpec(float, 0.0, 1.0),
    "trace_mode": SettingSpec(
        str,
        choices=("NONE", "SINGLE_SAMPLE", "MID_POINT", "MODIFIED_MID_POINT", "RK3"),
    ),
    "relax_enabled": SettingSpec(bool),
    "relax_strength": SettingSpec(float, 0.0, 1.0),
    "particle_radius_scale": SettingSpec(float, 0.5, 8.0),
    "cfl": SettingSpec(float, 0.1, 5.0),
    "min_substeps": SettingSpec(int, 1, 16),
    "max_substeps": SettingSpec(int, 1, 32),
    "pressure_solver": SettingSpec(str, choices=("JACOBI", "PCG")),
    "pressure_iterations": SettingSpec(int, 1, 4_000),
    "pressure_tolerance": SettingSpec(float, 0.0, 1.0),
    "viscosity_enabled": SettingSpec(bool),
    "viscosity_solver": SettingSpec(str, choices=("JACOBI", "PCG")),
    "viscosity_constant": SettingSpec(float, 0.0, 1_000_000.0),
    "viscosity_iterations": SettingSpec(int, 1, 1_000),
    "viscosity_tolerance": SettingSpec(float, 0.0, 1.0),
    "surface_tension_enabled": SettingSpec(bool),
    "surface_tension_coefficient": SettingSpec(float, 0.0, 10.0),
    "surface_tension_density": SettingSpec(float, 1.0e-6, 100_000.0),
    "surface_tension_band_voxels": SettingSpec(float, 0.5, 8.0),
    "surface_tension_safety": SettingSpec(float, 0.05, 1.0),
    "surface_tension_smoothing_iterations": SettingSpec(int, 0, 8),
    "surface_enabled": SettingSpec(bool),
    "surface_resolution_mode": SettingSpec(
        str, choices=("SIMULATION", "MULTIPLIER", "CUSTOM")
    ),
    "surface_resolution_multiplier": SettingSpec(float, 0.25, 8.0),
    "surface_splat_radius": SettingSpec(int, 1, 8),
    "surface_filter_iterations": SettingSpec(int, 0, 16),
    "surface_smooth_iterations": SettingSpec(int, 0, 16),
    "surface_smooth_strength": SettingSpec(float, 0.0, 1.0),
    "surface_update_cadence": SettingSpec(
        str, choices=("SUBSTEP", "FRAME", "MANUAL")
    ),
    "live_surface_publish_cadence": SettingSpec(int, 1, 8),
    "whitewater_enabled": SettingSpec(bool),
    "max_whitewater": SettingSpec(int, 1_024, 20_000_000),
    "ww_churn_decay": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_churn_gain": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_churn_threshold": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_emission_rate": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_surface_band": SettingSpec(float, 0.05, 64.0),
    "ww_min_speed": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_life_min": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_life_max": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_foam_band": SettingSpec(float, 0.05, 64.0),
    "ww_spray_min_speed": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_type_hysteresis": SettingSpec(float, 0.0, 0.95),
    "ww_foam_advection": SettingSpec(float, 0.0, 1.0),
    "ww_foam_preservation": SettingSpec(float, 0.0, 1.0),
    "ww_foam_density_influence": SettingSpec(float, 0.0, 100.0),
    "ww_foam_life_multiplier": SettingSpec(float, 0.01, 100.0),
    "ww_bubble_drag": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_bubble_buoyancy": SettingSpec(float, 0.0, 100.0),
    "ww_bubble_life_multiplier": SettingSpec(float, 0.01, 100.0),
    "ww_spray_drag": SettingSpec(float, 0.0, 1_000_000.0),
    "ww_spray_gravity_scale": SettingSpec(float, 0.0, 100.0),
    "ww_spray_life_multiplier": SettingSpec(float, 0.01, 100.0),
    "whitewater_output_mode": SettingSpec(
        str, choices=("COMBINED", "SEPARATE", "BOTH")
    ),
    "cache_compress": SettingSpec(bool),
    "checkpoint_interval": SettingSpec(int, 1, 100),
}

# The settings owned by the scene group. The group that declares a setting
# owns it: ``FLUIDSIM_PG_scene`` (``Scene.fluidsim``) holds exactly these names
# and ``FLUIDSIM_PG_domain`` (``Object.fluidsim_domain``) holds every other
# setting, with no name on both. Of the preset-eligible keys
# (``SETTING_SPECS``) only ``whitewater_output_mode``, ``cache_compress`` and
# ``checkpoint_interval`` are scene-owned: Output Publication and Production
# Cache Bake read them straight off ``scene.fluidsim``, so a preset dispatches
# them there and everything else to the active domain settings.
#
# This is the one pure-Python copy of that partition. It exists because
# ``presets/library.py`` must stay bpy-free, so it cannot read the class from
# ``properties.py``. ``blender/pending_settings`` (signatures, ``declaring_group``)
# and the staged sidebar read it from here.
# ``tests/runtime/test_pending_settings_registry.py`` pins it equal to the
# field set of ``FLUIDSIM_PG_scene``, so it cannot drift into a second opinion.
SCENE_OWNED_SETTINGS = frozenset({
    "ui_mode", "quality", "live", "playback_advanced_expanded",
    "max_frame_catchup", "domain_object", "capacity_show_alternatives",
    "surface_advanced_expanded", "liquid_view_mode", "show_surface",
    "show_particles", "show_whitewater", "particle_display_fraction",
    "particle_display_scale", "surface_attribute_mode",
    "particle_attribute_mode", "whitewater_attribute_mode",
    "whitewater_output_mode", "whitewater_display_scale",
    "whitewater_display_fraction", "surface_material", "particle_material",
    "whitewater_material", "foam_material", "bubble_material",
    "spray_material", "output_collection_name", "setup_mode",
    "product_preset", "active_preset_id", "cache_format", "bake_start",
    "bake_end", "checkpoint_interval", "bake_status", "bake_progress",
    "bake_message", "show_overlay", "report_path", "benchmark_frames",
    "benchmark_baseline", "cache_enabled", "cache_path", "cache_compress",
    "cache_playback_enabled", "cache_use_custom_range", "cache_frame_start",
    "cache_frame_end", "status", "stat_particles", "stat_voxels",
    "stat_whitewater", "stat_whitewater_foam", "stat_whitewater_bubbles",
    "stat_whitewater_spray", "stat_whitewater_invalid_types",
    "stat_whitewater_required", "stat_whitewater_admitted",
    "stat_whitewater_dropped", "stat_whitewater_policy",
    "stat_whitewater_status", "stat_step_ms", "stat_substeps",
    "stat_cfl_clamped", "stat_capillary_dt",
    "stat_capillary_substeps_required", "stat_capillary_clamped",
    "stat_substep_limiter", "stat_surface_tension_active_voxels",
    "stat_surface_tension_max_curvature",
    "stat_surface_tension_max_acceleration",
    "stat_surface_tension_temporary_bytes",
    "stat_surface_tension_submission_ms", "stat_surface_tension_status",
    "stat_surface_tension_message", "stat_native_solver_message",
})

_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "name",
    "description",
    "expected_scale",
    "performance_cost",
    "requires_reset",
    "settings",
    "source_version",
}


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"preset {label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class PresetV1:
    schema_version: int
    id: str
    name: str
    description: str
    expected_scale: str
    performance_cost: str
    requires_reset: bool
    settings: Mapping[str, Any]
    source_version: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PresetV1":
        if not isinstance(payload, Mapping):
            raise ValueError("preset must be a JSON object")
        unknown = set(payload) - _TOP_LEVEL_KEYS
        required = _TOP_LEVEL_KEYS - {"source_version"}
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                f"invalid preset keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        version = payload["schema_version"]
        if type(version) is not int or version != PRESET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported preset schema {version!r}; expected {PRESET_SCHEMA_VERSION}"
            )
        identifier = _required_string(payload["id"], "id").lower()
        if _ID_RE.fullmatch(identifier) is None:
            raise ValueError(
                "preset id must be 2-64 lowercase letters, numbers, or underscores"
            )
        requires_reset = payload["requires_reset"]
        if type(requires_reset) is not bool:
            raise ValueError("preset requires_reset must be a boolean")
        raw_settings = payload["settings"]
        if not isinstance(raw_settings, Mapping):
            raise ValueError("preset settings must be a JSON object")
        settings: dict[str, Any] = {}
        for key, value in raw_settings.items():
            if not isinstance(key, str) or key not in SETTING_SPECS:
                raise ValueError(f"unsupported setting {key!r}")
            settings[key] = SETTING_SPECS[key].validate(key, value)
        if not settings:
            raise ValueError("preset settings must not be empty")
        return cls(
            schema_version=PRESET_SCHEMA_VERSION,
            id=identifier,
            name=_required_string(payload["name"], "name"),
            description=_required_string(payload["description"], "description"),
            expected_scale=_required_string(payload["expected_scale"], "expected_scale"),
            performance_cost=_required_string(payload["performance_cost"], "performance_cost"),
            requires_reset=requires_reset,
            settings=settings,
            source_version=str(payload.get("source_version", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        # Revalidation keeps hand-constructed dataclass instances from bypassing
        # the public contract.
        payload = {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "expected_scale": self.expected_scale,
            "performance_cost": self.performance_cost,
            "requires_reset": self.requires_reset,
            "settings": dict(self.settings),
            "source_version": self.source_version,
        }
        validated = type(self).from_dict(payload)
        return {
            "schema_version": validated.schema_version,
            "id": validated.id,
            "name": validated.name,
            "description": validated.description,
            "expected_scale": validated.expected_scale,
            "performance_cost": validated.performance_cost,
            "requires_reset": validated.requires_reset,
            "settings": dict(validated.settings),
            "source_version": validated.source_version,
        }
