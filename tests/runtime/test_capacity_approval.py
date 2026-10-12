"""Capacity Approval: one planner-verified option is applied whole, or not at all.

``capacity_approval`` is a leaf module (no ``bpy``), so the transaction runs here
against plain stand-ins. The doubles model what makes the transaction necessary:
a real ``IntProperty`` / ``FloatProperty`` silently clamps an out-of-range write,
and a float property stores single precision.
"""

from __future__ import annotations

import ast
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import capacity_approval
from fluidSim_warp_blender51.blender.capacity_approval import CapacityApplyError, apply_suggestion
from fluidSim_warp_blender51.sim.quality_contract import (
    divisions_for_voxel_size,
    voxel_size_for_divisions,
)


class _Settings:
    """A PropertyGroup stand-in: INT/FLOAT writes clamp, FLOAT stores float32."""

    def __init__(self, ranges, **values):
        object.__setattr__(self, "_ranges", dict(ranges))
        object.__setattr__(
            self,
            "bl_rna",
            SimpleNamespace(
                properties={n: SimpleNamespace(type=kind) for n, (kind, _lo, _hi) in ranges.items()}
            ),
        )
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        spec = self._ranges.get(name)
        if spec is not None:
            kind, low, high = spec
            if kind == "INT":
                value = int(value)
            elif kind == "FLOAT":
                value = struct.unpack("f", struct.pack("f", float(value)))[0]
            value = min(max(value, low), high)
        object.__setattr__(self, name, value)


RANGES = {
    "leaf_capacity": ("INT", 64, 262_144),
    "max_particles": ("INT", 1024, 50_000_000),
    "voxel_size": ("FLOAT", 0.001, 10.0),
}


def _settings(**overrides):
    values = {"leaf_capacity": 8192, "max_particles": 2_000_000, "voxel_size": 0.1}
    values.update(overrides)
    return _Settings(RANGES, **values)


def _domain(**overrides):
    values = {
        "location": (0.0, 0.0, 0.0),
        "dimensions": (4.0, 4.0, 4.0),
        "rotation_euler": (0.0, 0.0, 0.0),
        "delta_rotation_euler": (0.0, 0.0, 0.0),
        "parent": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


AUTHORED = SimpleNamespace(domain_min=(0.0, 0.0, 0.0), domain_max=(10.0, 10.0, 10.0))


def _apply(settings, domain, changes, **kwargs):
    kwargs.setdefault("build_settings", lambda: AUTHORED)
    return apply_suggestion(settings, domain, changes, **kwargs)


# --- settings ---------------------------------------------------------------


def test_every_change_is_written_and_named_in_order():
    settings = _settings()

    changed = _apply(settings, _domain(), {"max_particles": 4_000_000, "leaf_capacity": 16_384})

    assert changed == ["max_particles", "leaf_capacity"]
    assert (settings.max_particles, settings.leaf_capacity) == (4_000_000, 16_384)


def test_an_integer_setting_is_written_as_an_integer():
    settings = _settings()

    _apply(settings, _domain(), {"max_particles": 4_000_000.0})

    assert settings.max_particles == 4_000_000
    assert isinstance(settings.max_particles, int)


def test_a_float_setting_may_round_to_single_precision():
    settings = _settings()

    _apply(settings, _domain(), {"voxel_size": 0.123456789})

    assert settings.voxel_size == pytest.approx(0.123456789, rel=1e-6)


def test_the_callers_change_map_is_not_consumed():
    changes = {"max_particles": 4_000_000, "domain_min": (0, 0, 0), "domain_max": (2, 2, 2)}

    _apply(_settings(), _domain(), changes)

    assert set(changes) == {"max_particles", "domain_min", "domain_max"}


def test_a_value_the_range_clamps_is_refused_and_everything_is_restored():
    settings = _settings()

    with pytest.raises(CapacityApplyError, match=r"'leaf_capacity' was stored as 64 instead of 32"):
        _apply(settings, _domain(), {"max_particles": 4_000_000, "leaf_capacity": 32})

    # The write that succeeded before the clamped one is undone too.
    assert (settings.max_particles, settings.leaf_capacity) == (2_000_000, 8192)


def test_an_unknown_setting_is_refused_and_earlier_writes_are_restored():
    settings = _settings()

    with pytest.raises(CapacityApplyError, match="cannot apply setting 'bogus'"):
        _apply(settings, _domain(), {"max_particles": 4_000_000, "bogus": 1})

    assert settings.max_particles == 2_000_000


def test_settings_alone_need_no_domain_object():
    settings = _settings()

    assert _apply(settings, None, {"max_particles": 4_000_000}) == ["max_particles"]
    assert settings.max_particles == 4_000_000


# --- domain bounds ----------------------------------------------------------


def test_bounds_resize_the_domain_and_refresh_the_view_once():
    domain = _domain()
    refreshed = []

    changed = _apply(
        _settings(),
        domain,
        {"max_particles": 4_000_000, "domain_min": (-1, -2, 0), "domain_max": (3, 2, 4)},
        on_transform_changed=lambda: refreshed.append(domain.dimensions),
    )

    assert changed == ["max_particles", "domain bounds"]
    assert domain.location == (1.0, 0.0, 2.0)
    assert domain.dimensions == (4.0, 4.0, 4.0)
    assert refreshed == [(4.0, 4.0, 4.0)]  # seen after the transform was written


def test_one_given_bound_is_completed_from_the_authored_settings():
    domain = _domain()

    _apply(_settings(), domain, {"domain_max": (2.0, 4.0, 6.0)})

    assert domain.location == (1.0, 2.0, 3.0)
    assert domain.dimensions == (2.0, 4.0, 6.0)

    other = _domain()
    _apply(_settings(), other, {"domain_min": (6.0, 6.0, 6.0)})
    assert other.location == (8.0, 8.0, 8.0)
    assert other.dimensions == (4.0, 4.0, 4.0)


@pytest.mark.parametrize(
    ("domain_overrides", "message"),
    [
        ({"parent": object()}, "requires an unparented domain"),
        ({"rotation_euler": (0.0, 0.0, 0.5)}, "requires an axis-aligned domain"),
        ({"delta_rotation_euler": (0.0, 0.25, 0.0)}, "requires an axis-aligned domain"),
    ],
)
def test_a_domain_that_cannot_be_resized_is_refused_and_untouched(domain_overrides, message):
    settings = _settings()
    domain = _domain(**domain_overrides)

    with pytest.raises(CapacityApplyError, match=message):
        _apply(
            settings,
            domain,
            {"max_particles": 4_000_000, "domain_min": (0, 0, 0), "domain_max": (2, 2, 2)},
        )

    assert settings.max_particles == 2_000_000
    assert (domain.location, domain.dimensions) == ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))


def test_a_full_turn_of_rotation_still_counts_as_axis_aligned():
    domain = _domain(rotation_euler=(0.0, 0.0, 2 * 3.141592653589793))

    _apply(_settings(), domain, {"domain_min": (0, 0, 0), "domain_max": (2, 2, 2)})

    assert domain.dimensions == (2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    "bounds",
    [
        ((0, 0, 0), (0, 2, 2)),  # zero width
        ((0, 0, 0), (2, -1, 2)),  # inverted
    ],
)
def test_bounds_with_no_volume_are_refused(bounds):
    domain = _domain()

    with pytest.raises(CapacityApplyError, match="Suggested domain bounds are not valid"):
        _apply(_settings(), domain, {"domain_min": bounds[0], "domain_max": bounds[1]})

    assert domain.dimensions == (4.0, 4.0, 4.0)


def test_bounds_without_a_domain_object_are_refused_and_settings_restored():
    settings = _settings()

    with pytest.raises(CapacityApplyError, match="no domain object to resize"):
        _apply(
            settings,
            None,
            {"max_particles": 4_000_000, "domain_min": (0, 0, 0), "domain_max": (2, 2, 2)},
        )

    assert settings.max_particles == 2_000_000


def test_a_failure_after_the_resize_puts_the_transform_back_too():
    settings = _settings()
    domain = _domain()

    def refresh_fails():
        raise RuntimeError("view layer is gone")

    with pytest.raises(CapacityApplyError, match="view layer is gone"):
        _apply(
            settings,
            domain,
            {"max_particles": 4_000_000, "domain_min": (0, 0, 0), "domain_max": (2, 2, 2)},
            on_transform_changed=refresh_fails,
        )

    assert settings.max_particles == 2_000_000
    assert (domain.location, domain.dimensions) == ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))


class _MeshDomain:
    """An object whose mesh is not centred on its origin (B54).

    Models Blender closely enough for an axis-aligned, unparented object: the
    world AABB is ``location + delta_location + scale * delta_scale * local``,
    ``dimensions`` rescales against the local bound box, and ``matrix_world``
    follows the transform (as it does after the view layer updates).
    """

    def __init__(self, local_min, local_max, location=(0.0, 0.0, 0.0),
                 delta_location=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                 delta_scale=(1.0, 1.0, 1.0)):
        self.local_min, self.local_max = tuple(local_min), tuple(local_max)
        self.location = tuple(location)
        self.delta_location = tuple(delta_location)
        self.scale = tuple(scale)
        self.delta_scale = tuple(delta_scale)
        self.rotation_euler = (0.0, 0.0, 0.0)
        self.delta_rotation_euler = (0.0, 0.0, 0.0)
        self.parent = None

    @property
    def bound_box(self):
        lo, hi = self.local_min, self.local_max
        return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])]

    def _effective_scale(self):
        return tuple(s * d for s, d in zip(self.scale, self.delta_scale))

    @property
    def dimensions(self):
        return tuple(abs(s) * (hi - lo) for s, lo, hi in zip(
            self._effective_scale(), self.local_min, self.local_max))

    @dimensions.setter
    def dimensions(self, value):
        self.scale = tuple(
            (1.0 if s >= 0.0 else -1.0) * v / ((hi - lo) * abs(d))
            for v, s, d, lo, hi in zip(value, self.scale, self.delta_scale,
                                       self.local_min, self.local_max))

    @property
    def matrix_world(self):
        s = self._effective_scale()
        t = tuple(a + b for a, b in zip(self.location, self.delta_location))
        return [[s[0], 0.0, 0.0, t[0]], [0.0, s[1], 0.0, t[1]],
                [0.0, 0.0, s[2], t[2]], [0.0, 0.0, 0.0, 1.0]]

    def world_bounds(self):
        m = self.matrix_world
        points = [tuple(m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3]
                        for r in range(3)) for p in self.bound_box]
        return (tuple(min(p[i] for p in points) for i in range(3)),
                tuple(max(p[i] for p in points) for i in range(3)))


@pytest.mark.parametrize(
    "domain_kwargs",
    [
        # Mesh with transforms applied: origin at world 0, geometry at 2..6.
        {"local_min": (2.0, 2.0, 0.0), "local_max": (6.0, 6.0, 4.0)},
        # Corner origin plus a delta offset and a delta scale.
        {"local_min": (0.0, 0.0, 0.0), "local_max": (1.0, 2.0, 0.5),
         "location": (1.0, -1.0, 0.0), "delta_location": (0.5, 0.0, -0.25),
         "scale": (4.0, 2.0, 8.0), "delta_scale": (1.0, 1.0, 1.0)},
        # Mirrored (negative scale) on one axis.
        {"local_min": (-1.0, 0.0, -1.0), "local_max": (3.0, 4.0, 3.0),
         "scale": (-1.0, 1.0, 1.0)},
    ],
)
def test_bounds_land_where_the_option_says_for_an_off_centre_mesh(domain_kwargs):
    domain = _MeshDomain(**domain_kwargs)
    requested = ((2.0, 2.0, 0.5), (4.0, 5.0, 2.0))

    changed = _apply(
        _settings(), domain,
        {"domain_min": requested[0], "domain_max": requested[1]},
        on_transform_changed=lambda: None)

    assert changed == ["domain bounds"]
    lo, hi = domain.world_bounds()
    assert lo == pytest.approx(requested[0], abs=1e-9)
    assert hi == pytest.approx(requested[1], abs=1e-9)


def test_a_resize_whose_world_bounds_do_not_match_is_rolled_back():
    class _StaleMatrixDomain(_MeshDomain):
        _frozen = None

        @property
        def matrix_world(self):
            # Blender only re-evaluates matrix_world on a view-layer update; a
            # refresh that did not take leaves the old transform in place.
            if self._frozen is None:
                self._frozen = _MeshDomain.matrix_world.fget(self)
            return self._frozen

    domain = _StaleMatrixDomain((2.0, 2.0, 0.0), (6.0, 6.0, 4.0))
    domain.matrix_world  # freeze the authored transform
    settings = _settings()

    with pytest.raises(CapacityApplyError, match="landed at"):
        _apply(settings, domain,
               {"max_particles": 4_000_000, "domain_min": (2.0, 2.0, 0.0),
                "domain_max": (4.0, 4.0, 2.0)},
               on_transform_changed=lambda: None)

    assert settings.max_particles == 2_000_000
    assert (domain.location, domain.dimensions) == ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))


def test_unreadable_authored_settings_roll_the_option_back():
    settings = _settings()

    def broken():
        raise ValueError("no solver settings")

    with pytest.raises(CapacityApplyError, match="no solver settings"):
        _apply(
            settings,
            _domain(),
            {"max_particles": 4_000_000, "domain_max": (2, 2, 2)},
            build_settings=broken,
        )

    assert settings.max_particles == 2_000_000


def test_the_error_keeps_its_cause():
    with pytest.raises(CapacityApplyError) as excinfo:
        _apply(_settings(), _domain(), {"bogus": 1})

    assert isinstance(excinfo.value.__cause__, RuntimeError)


# --- the dialog's helpers ---------------------------------------------------


def test_find_suggestion_matches_by_key_and_tolerates_a_missing_plan():
    safe = SimpleNamespace(key="safe")
    plan = SimpleNamespace(suggestions=(SimpleNamespace(key="balanced"), safe))

    assert capacity_approval.find_suggestion(plan, "safe") is safe
    assert capacity_approval.find_suggestion(plan, "other") is None
    assert capacity_approval.find_suggestion(None, "safe") is None
    assert capacity_approval.find_suggestion(SimpleNamespace(), "safe") is None


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (4096, "4,096"),
        (2_000_000, "2,000,000"),
        (0.1, "0.1"),
        (0.123456789, "0.123457"),
        ((0.5, 12.0, -3.14159), "(0.5, 12, -3.142)"),
        ([1, 2], "(1, 2)"),
        (None, "None"),
        ("auto", "auto"),
    ],
)
def test_values_are_formatted_for_the_dialog(value, text):
    assert capacity_approval.format_value(value) == text


def test_the_module_is_a_leaf_that_never_imports_blender_or_the_add_on():
    tree = ast.parse(Path(capacity_approval.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative (add-on) imports"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "math"}


# --- derived voxel_size (resolution divisions over the domain's longest axis) ---


class _DerivedSettings:
    """Models properties.py: voxel_size = longest axis / whole divisions."""

    def __init__(self, domain, divisions):
        object.__setattr__(self, "_domain", domain)
        object.__setattr__(self, "resolution_divisions", divisions)
        object.__setattr__(self, "bl_rna", SimpleNamespace(properties={}))

    @property
    def voxel_size(self):
        return voxel_size_for_divisions(self.resolution_divisions, max(self._domain.dimensions))

    @voxel_size.setter
    def voxel_size(self, value):
        object.__setattr__(
            self,
            "resolution_divisions",
            divisions_for_voxel_size(value, max(self._domain.dimensions)),
        )


def test_a_lattice_voxel_round_trips_through_a_derived_voxel_setting():
    domain = _domain(dimensions=(10.0, 4.0, 4.0))
    settings = _DerivedSettings(domain, 100)
    target = voxel_size_for_divisions(77, 10.0)

    changed = _apply(settings, domain, {"voxel_size": target})

    assert changed == ["voxel_size"]
    assert settings.resolution_divisions == 77


def test_bounds_and_voxel_are_costed_against_the_resized_domain():
    domain = _domain(dimensions=(10.0, 10.0, 10.0))
    settings = _DerivedSettings(domain, 100)
    # Costed against the NEW 4 m longest axis; against the old 10 m it would
    # round to a different divisions count and fail the strict read-back.
    target = voxel_size_for_divisions(30, 4.0)

    changed = _apply(
        settings,
        domain,
        {"voxel_size": target, "domain_min": (-2, -2, -2), "domain_max": (2, 2, 2)},
    )

    assert changed == ["voxel_size", "domain bounds"]
    assert settings.resolution_divisions == 30
    assert settings.voxel_size == target


def test_a_failed_bounds_and_voxel_option_restores_divisions_and_transform():
    domain = _domain(dimensions=(10.0, 10.0, 10.0))
    settings = _DerivedSettings(domain, 100)
    before = settings.voxel_size

    def refresh_fails_after_resize(calls=[]):
        calls.append(1)
        if len(calls) == 1:
            return
        raise RuntimeError("view layer is gone")

    with pytest.raises(CapacityApplyError):
        _apply(
            settings,
            domain,
            {"voxel_size": 0.1, "domain_min": (-2, -2, -2), "domain_max": (2, 2, 2), "bogus": 1},
            on_transform_changed=refresh_fails_after_resize,
        )

    assert settings.resolution_divisions == 100
    assert settings.voxel_size == before
    assert domain.dimensions == (10.0, 10.0, 10.0)
