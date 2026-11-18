"""Contracts for the tsunami_parthenon demo's temple and water geometry."""

from collections import Counter

import numpy as np
import pytest

from tools.demos import tsunami_parthenon as tp


def _shells(mesh):
    """Split faces into connected components (shared vertices)."""
    parent = list(range(len(mesh.verts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in mesh.faces:
        for v in face[1:]:
            parent[find(v)] = find(face[0])
    groups = {}
    for face in mesh.faces:
        groups.setdefault(find(face[0]), []).append(face)
    return list(groups.values())


@pytest.fixture(scope="module")
def meshes():
    return tp.collider_meshes()


def test_scene_has_the_expected_parts(meshes):
    names = [m.name for m in meshes]
    assert names == [
        "Tsunami Terrace", "Temple Step 1", "Temple Step 2", "Temple Step 3",
        "Temple Columns", "Temple Cella", "Temple Entablature", "Temple Roof"]
    assert len(set(names)) == len(names)


def test_peristyle_is_the_parthenon_8_by_17_ring():
    centers = tp.column_centers()
    assert len(centers) == 2 * 17 + 2 * (8 - 2)
    assert len({tuple(np.round(c, 6)) for c in centers}) == len(centers)
    assert len(_shells(tp.columns_mesh())) == len(centers)


def test_every_mesh_is_closed_and_consistently_wound(meshes):
    for mesh in meshes:
        directed = Counter()
        for face in mesh.faces:
            for k, a in enumerate(face):
                directed[(a, face[(k + 1) % len(face)])] += 1
        assert max(directed.values()) == 1, f"{mesh.name}: repeated directed edge"
        for a, b in directed:
            assert directed[(b, a)] == 1, f"{mesh.name}: open edge {a}-{b}"


def test_every_shell_faces_outward(meshes):
    for mesh in meshes:
        for shell in _shells(mesh):
            assert tp._signed_volume(mesh.verts, shell) > 0.0, mesh.name


def test_no_degenerate_faces(meshes):
    for mesh in meshes:
        for face in mesh.faces:
            assert len(set(face)) == len(face), mesh.name
            area = sum(np.linalg.norm(np.cross(
                mesh.verts[face[k]] - mesh.verts[face[0]],
                mesh.verts[face[k + 1]] - mesh.verts[face[0]]))
                for k in range(1, len(face) - 1))
            assert area > 1e-9, mesh.name


def test_shells_within_one_mesh_do_not_overlap(meshes):
    for mesh in meshes:
        boxes = []
        for shell in _shells(mesh):
            idx = sorted({v for face in shell for v in face})
            pts = mesh.verts[idx]
            boxes.append((pts.min(axis=0), pts.max(axis=0)))
        for i, (lo_a, hi_a) in enumerate(boxes):
            for lo_b, hi_b in boxes[i + 1:]:
                overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
                assert (overlap <= 0.0).any(), mesh.name


def test_columns_are_at_least_three_voxels_thick_and_leave_room_to_pass():
    flats = 2 * tp.COLUMN_R_TOP * np.cos(np.pi / tp.COLUMN_SIDES)
    assert flats >= tp.MIN_THICKNESS - 1e-9
    assert tp.COLUMN_SPACING - 2 * tp.COLUMN_R_BOTTOM >= 2.5 * tp.VOXEL


def test_walls_beams_and_roof_are_at_least_three_voxels_thick():
    assert tp.CELLA_WALL >= tp.MIN_THICKNESS - 1e-9
    assert tp.ENTABLATURE_BAND >= tp.MIN_THICKNESS
    assert tp.ENTABLATURE_HEIGHT >= tp.MIN_THICKNESS
    assert tp.ROOF_EAVE >= tp.MIN_THICKNESS - 1e-9
    # The door and the wall stubs beside it must survive the voxel grid.
    stub = tp.CELLA_HALF[1] - tp.CELLA_WALL - tp.CELLA_DOOR_HALF
    assert stub >= tp.MIN_THICKNESS
    assert 2 * tp.CELLA_DOOR_HALF >= 6 * tp.VOXEL
    # The terrace is thick everywhere, including the toe of the ramp.
    terrace = tp.terrace_mesh()
    assert terrace.verts[:, 2].min() == pytest.approx(tp.BASE_Z)
    assert 0.0 - tp.BASE_Z >= tp.MIN_THICKNESS


def test_temple_sits_inside_the_domain_with_room_around_it(meshes):
    lo, hi = np.array(tp.DOMAIN_LO), np.array(tp.DOMAIN_HI)
    for mesh in meshes:
        assert (mesh.verts[:, 0] >= lo[0] - 1e-9).all(), mesh.name
        assert (mesh.verts[:, 0] <= hi[0] + 1e-9).all(), mesh.name
        assert (mesh.verts[:, 1] >= lo[1] - 1e-9).all(), mesh.name
        assert (mesh.verts[:, 1] <= hi[1] + 1e-9).all(), mesh.name
        assert mesh.verts[:, 2].max() <= hi[2] - 1.0, mesh.name
    assert tp.TEMPLE_FRONT_X > tp.RAMP_END_X
    assert tp.TEMPLE_REAR_X <= tp.DOMAIN_HI[0]


def test_parts_rest_on_each_other_with_a_hair_of_overlap():
    assert tp.COLUMN_BASE_Z < tp.STYLOBATE_Z
    assert tp.ENTABLATURE_Z0 < tp.COLUMN_TOP_Z
    assert tp.ROOF_Z0 < tp.ENTABLATURE_Z1
    assert tp.step_tops() == tuple(sorted(tp.step_tops(), reverse=True))
    assert tp.step_tops()[-1] > tp.TERRACE_Z


def test_roof_overhangs_the_beams_and_the_cella_fits_inside_the_colonnade():
    entablature = tp.entablature_mesh()
    assert tp.ROOF_HALF_X > entablature.verts[:, 0].max() - tp.TEMPLE_CENTER[0]
    assert tp.ROOF_HALF_Y > entablature.verts[:, 1].max() - tp.TEMPLE_CENTER[1]
    inner_x = (tp.COLUMNS_X - 1) * tp.COLUMN_SPACING / 2 - tp.COLUMN_R_BOTTOM
    inner_y = (tp.COLUMNS_Y - 1) * tp.COLUMN_SPACING / 2 - tp.COLUMN_R_BOTTOM
    assert tp.CELLA_HALF[0] < inner_x - 3 * tp.VOXEL
    assert tp.CELLA_HALF[1] < inner_y - 3 * tp.VOXEL


def test_the_wave_is_taller_than_the_temple():
    assert tp.WAVE_HI[2] > tp.TEMPLE_TOP_Z
    assert tp.TEMPLE_TOP_Z < tp.DOMAIN_HI[2] - 1.0


def test_water_sits_upstream_of_the_temple_and_inside_the_domain():
    for lo, hi in ((tp.WAVE_LO, tp.WAVE_HI), (tp.FORESHORE_LO, tp.FORESHORE_HI)):
        assert (np.array(lo) >= np.array(tp.DOMAIN_LO) - 1e-9).all()
        assert (np.array(hi) <= np.array(tp.DOMAIN_HI) + 1e-9).all()
        assert hi[0] <= tp.RAMP_START_X + 1e-9
    assert tp.expected_particles() < tp.MAX_PARTICLES


def test_the_wave_is_much_bigger_than_the_dam_break_reservoir():
    from tools.demos import dam_break_cove as cove

    dam = float(np.prod(np.subtract(cove.RESERVOIR_HI, cove.RESERVOIR_LO)))
    wave = float(np.prod(np.subtract(tp.WAVE_HI, tp.WAVE_LO)))
    assert wave > 5 * dam


def test_analysis_helpers():
    cx, cy = tp.TEMPLE_CENTER
    inside = [cx, cy, 1.0]
    outside = [cx, cy, 2.0]
    assert tp.colonnade_fill([inside, outside]) == 1
    assert tp.front_x([[1, 0, 0], [3, 0, 0]], quantile=1.0) == 3.0
    assert tp.fraction_past([[1, 0, 0], [3, 0, 0]], 2.0) == 0.5
    assert tp.front_x([]) == 0.0
    assert tp.colonnade_fill(np.zeros((0, 3))) == 0
