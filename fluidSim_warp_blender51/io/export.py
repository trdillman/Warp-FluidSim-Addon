"""Optional export tooling: surface OBJ and particle PLY snapshots."""

from __future__ import annotations

import os
import tempfile

import numpy as np


def _host_frame(solver):
    from ..sim.frame_output import HostFrame
    return solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)


def _require_existing_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Output parent directory does not exist: {parent}")


def _flush_and_sync(fh) -> None:
    fh.flush()
    if hasattr(os, "fsync"):
        os.fsync(fh.fileno())


def _surface_payload(solver, host):
    verts, tris = host.surface_mesh
    verts = np.asarray(verts)
    tris = np.asarray(tris)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("surface vertices must have shape (N, 3)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("surface triangles must have shape (N, 3)")
    try:
        finite = np.isfinite(verts).all()
    except TypeError as exc:
        raise ValueError("surface vertices must be finite numbers") from exc
    if not finite:
        raise ValueError("surface vertices must be finite")
    metadata = solver.surface.metadata()
    header = (
        float(metadata["voxel_size"]),
        int(solver.settings.surface_resolution_mode),
        float(solver.settings.surface_resolution_multiplier),
        int(solver.settings.surface_smooth_iterations),
    )
    return verts.copy(), tris.copy(), header


def _write_surface_obj(path: str, payload) -> None:
    verts, tris, header = payload
    voxel_size, resolution_mode, resolution_multiplier, smooth_iterations = header
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# FluidSim Warp surface export\n")
        fh.write(f"# surface_voxel_size {voxel_size:.8g}\n")
        fh.write(f"# surface_resolution_mode {resolution_mode}\n")
        fh.write(
            "# surface_resolution_multiplier "
            f"{resolution_multiplier:.8g}\n"
        )
        fh.write(f"# surface_smooth_iterations {smooth_iterations}\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        _flush_and_sync(fh)


def _points_payload(positions: np.ndarray,
                    velocities: np.ndarray | None = None):
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    with np.errstate(over="ignore", invalid="ignore"):
        positions = np.ascontiguousarray(positions, dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")

    has_vel = velocities is not None
    if has_vel:
        velocities = np.asarray(velocities)
        if velocities.ndim != 2 or velocities.shape != positions.shape:
            raise ValueError("velocities must match positions shape (N, 3)")
        with np.errstate(over="ignore", invalid="ignore"):
            velocities = np.ascontiguousarray(velocities, dtype=np.float32)
        if not np.isfinite(velocities).all():
            raise ValueError("velocities must be finite")
        data = np.ascontiguousarray(np.hstack([positions, velocities]))
    else:
        data = positions
    return len(positions), has_vel, data


def _write_points_ply(path: str, payload) -> None:
    count, has_vel, data = payload
    with open(path, "wb") as fh:
        header = ["ply", "format binary_little_endian 1.0",
                  f"element vertex {count}",
                  "property float x", "property float y", "property float z"]
        if has_vel:
            header += ["property float vx", "property float vy",
                       "property float vz"]
        header += ["end_header"]
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.tobytes())
        _flush_and_sync(fh)


def _transaction_path(final_path: str, suffix: str) -> str:
    parent = os.path.dirname(os.path.abspath(final_path))
    basename = os.path.basename(final_path)
    fd, path = tempfile.mkstemp(prefix=f".{basename}.", suffix=suffix, dir=parent)
    os.close(fd)
    return path


def _remove_if_present(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _publish_bundle(artifacts) -> list[str]:
    """Publish a prevalidated sibling bundle with in-process rollback.

    Every payload is durably staged before any final path is replaced. A caught
    publication failure restores the prior bundle, but separate filesystem
    renames cannot provide crash-atomic multi-file publication.
    """
    staged = {}
    backups = {}
    published = set()
    retained_backups = set()
    finals = [path for path, _, _ in artifacts]
    try:
        for final_path, writer, payload in artifacts:
            stage_path = _transaction_path(final_path, ".stage")
            staged[final_path] = stage_path
            writer(stage_path, payload)

        for final_path in finals:
            if os.path.exists(final_path):
                backup_path = _transaction_path(final_path, ".backup")
                _remove_if_present(backup_path)
                backups[final_path] = backup_path
                os.replace(final_path, backup_path)

        for final_path in finals:
            os.replace(staged[final_path], final_path)
            published.add(final_path)
        return finals
    except Exception as publish_error:
        rollback_errors = []
        for final_path in reversed(finals):
            backup_path = backups.get(final_path)
            try:
                if backup_path is not None and os.path.exists(backup_path):
                    os.replace(backup_path, final_path)
                elif final_path in published:
                    _remove_if_present(final_path)
            except OSError as rollback_error:
                rollback_errors.append((final_path, rollback_error))
                if backup_path is not None and os.path.exists(backup_path):
                    retained_backups.add(backup_path)
        if rollback_errors:
            details = "; ".join(
                f"{path}: {error}" for path, error in rollback_errors
            )
            raise RuntimeError(
                "Export publication failed and rollback was incomplete; "
                f"recoverable backups were retained ({details})"
            ) from publish_error
        raise
    finally:
        for owned_path in staged.values():
            _remove_if_present(owned_path)
        for owned_path in backups.values():
            if owned_path not in retained_backups:
                _remove_if_present(owned_path)


def export_surface_obj(solver, path: str, *, host=None) -> str:
    _require_existing_parent(path)
    payload = _surface_payload(solver, host or _host_frame(solver))
    _write_surface_obj(path, payload)
    return path


def export_points_ply(positions: np.ndarray, path: str,
                      velocities: np.ndarray | None = None) -> str:
    _require_existing_parent(path)
    payload = _points_payload(positions, velocities)
    _write_points_ply(path, payload)
    return path


def export_frame(solver, base_path: str) -> list:
    """Export one validated frame bundle with rollback on ordinary failures."""
    _require_existing_parent(base_path)
    host = _host_frame(solver)
    artifacts = []
    if solver.settings.surface_enabled:
        artifacts.append((
            base_path + "_surface.obj",
            _write_surface_obj,
            _surface_payload(solver, host),
        ))
    pos = host.positions
    if len(pos):
        artifacts.append((
            base_path + "_particles.ply",
            _write_points_ply,
            _points_payload(pos, host.velocities),
        ))
    ww = host.whitewater
    if len(ww):
        artifacts.append((
            base_path + "_whitewater.ply",
            _write_points_ply,
            _points_payload(ww),
        ))
    return _publish_bundle(artifacts)
