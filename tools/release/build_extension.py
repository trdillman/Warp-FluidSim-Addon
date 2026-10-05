"""Verify, build, inspect, and smoke-test FluidSim Blender extensions."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib

TOOL_ROOT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
REQUIRED_EXTENSION_ID = "fluidsim_warp"
REQUIRED_LICENSE = ["SPDX:GPL-3.0-or-later"]
REQUIRED_PLATFORMS = ("windows-x64", "linux-x64")
REQUIRED_WARP_VERSION = "1.17.0"
REQUIRED_WARP_DISTRIBUTION = "1.17.0+cu12"
REQUIRED_BLENDER_MIN = "5.2.0"
BUNDLED_WHEEL_DIR = "wheels"
RELEASE_ONLY_FILES = {"pyproject.toml", "requirements-dev.txt"}
RELEASE_ONLY_RELATIVE_PATHS: set[str] = set()
RELEASE_ONLY_PATHS = {
    "tests",
    "demo_outputs",
    "artifacts",
    "build",
    "dist",
    ".git",
    ".github",
}
Runner = Callable[..., Any]
_BLENDER_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _tool_root_for(addon_root: Path) -> Path:
    """Return the sibling developer-tool directory for an add-on source tree."""
    addon_root = Path(addon_root).resolve()
    if addon_root == ROOT.resolve():
        return TOOL_ROOT
    return addon_root.parent / "tools" / "release"


def _looks_like_blender_file(payload: bytes) -> bool:
    return payload.startswith((b"BLENDER", _BLENDER_ZSTD_MAGIC))


@dataclass(frozen=True)
class SourceVerification:
    """Result of verifying release metadata and bundled runtime locks."""

    extension_id: str
    version: str
    platforms: tuple[str, ...]
    bundled_warp_wheels: tuple[str, ...]
    wheel_platforms: tuple[tuple[str, str], ...] = ()
    wheel_sha256: tuple[tuple[str, str], ...] = ()

    def wheel_for_platform(self, platform: str) -> str:
        matches = [name for name, owner in self.wheel_platforms if owner == platform]
        if len(matches) != 1:
            raise RuntimeError(f"expected one locked Warp wheel for {platform}, found {matches}")
        return matches[0]

    @property
    def release_ready(self) -> bool:
        return True


@dataclass(frozen=True)
class ArtifactRecord:
    """Verified split-platform package metadata."""

    path: Path
    platform: str
    runtime_dependency: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def _run(command: list[str], cwd: Path, *, env: Mapping[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _blender_process_environment(
    source: Mapping[str, str],
    *,
    os_name: str | None = None,
    compiled_platform: str | None = None,
) -> dict[str, str]:
    """Return an environment with truthful Windows architecture metadata.

    Blender's extension installer derives its platform suffix through
    ``platform.machine()``. Managed shells can omit
    ``PROCESSOR_ARCHITECTURE``, which makes that call return an empty string
    even though Python's compiled platform remains authoritative.
    """
    env = {str(key): str(value) for key, value in source.items()}
    active_os = os.name if os_name is None else str(os_name)
    if active_os != "nt" or env.get("PROCESSOR_ARCHITECTURE"):
        return env
    platform_tag = str(
        sysconfig.get_platform()
        if compiled_platform is None
        else compiled_platform
    ).lower().replace("_", "-")
    architecture = {
        "win-amd64": "AMD64",
        "win-arm64": "ARM64",
        "win32": "x86",
    }.get(platform_tag)
    if architecture is None:
        raise RuntimeError(
            "cannot derive Blender Windows architecture from compiled "
            f"platform {platform_tag!r}"
        )
    env["PROCESSOR_ARCHITECTURE"] = architecture
    return env


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"could not read TOML metadata {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _assigned_literal(path: Path, name: str) -> Any:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"could not parse {path}: {exc}") from exc
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f"{name} in {path} must be a literal") from exc
    raise RuntimeError(f"{name} is not assigned in {path}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _collision_guard(root: Path) -> dict[str, str | None]:
    """Map each protected owner to its stale digest; ``None`` marks a retired owner.

    A retired owner was deleted on purpose, so it must stay absent: the stale
    candidate would otherwise bring it back.
    """
    guard_path = _tool_root_for(root) / "candidate_collision_guard_1_0_253.json"
    guard = _read_json(guard_path)
    candidate_sha256 = guard.get("candidate_sha256")
    if (
        not isinstance(candidate_sha256, str)
        or len(candidate_sha256) != 64
        or any(char not in "0123456789abcdef" for char in candidate_sha256)
    ):
        raise RuntimeError("candidate collision guard has an invalid archive SHA-256")
    forbidden = guard.get("forbidden_sha256")
    if not isinstance(forbidden, dict) or not forbidden:
        raise RuntimeError("candidate collision guard must protect at least one owner")
    retired = guard.get("retired_owners", [])
    if not isinstance(retired, list) or not set(retired) <= forbidden.keys():
        raise RuntimeError("candidate collision guard retired_owners must list protected owners")
    result: dict[str, str | None] = {}
    for relative, digest in forbidden.items():
        if not isinstance(relative, str):
            raise TypeError("candidate collision guard paths must be strings")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or str(path) != relative:
            raise RuntimeError(f"unsafe candidate collision guard path: {relative!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RuntimeError(f"invalid stale candidate SHA-256 for {relative}")
        result[relative] = None if relative in retired else digest
    return result


def _verify_source_collision_guard(root: Path, forbidden: Mapping[str, str | None]) -> None:
    for relative, stale_sha256 in forbidden.items():
        path = root / PurePosixPath(relative)
        if stale_sha256 is None:
            if path.exists():
                raise RuntimeError(f"retired integration owner reappeared: {relative}")
            continue
        if not path.is_file():
            raise RuntimeError(f"protected integration owner is missing: {relative}")
        if _sha256_path(path) == stale_sha256:
            raise RuntimeError(
                f"stale candidate implementation replaced adjudicated owner: {relative}"
            )


def _locked_files(lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("tools/release/wheels.lock.json must contain a non-empty files object")
    result: dict[str, dict[str, Any]] = {}
    for filename, metadata in files.items():
        if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
            raise RuntimeError(f"invalid locked wheel filename: {filename!r}")
        if not isinstance(metadata, dict):
            raise TypeError(f"invalid lock metadata for {filename}")
        result[filename] = metadata
    return result


def verify_source_tree(root: Path) -> SourceVerification:
    """Validate release identity and the bundled, hash-locked Warp wheels."""
    root = Path(root).resolve()
    required = (
        "__init__.py",
        "blender_manifest.toml",
        "version.py",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "sim/surface_tension.py",
        "sim/surface_tension_math.py",
        "resources/presets/water_surface_tension.json",
        "resources/presets/stylized_high_tension.json",
        "sim/effects/schema.py",
        "sim/effects/runtime.py",
        "sim/effects/package.py",
        "sim/effects/templates.py",
        "blender/effects.py",
        "blender/effect_ui.py",
        "blender/effect_workspace.py",
        "resources/effects/vortex_force.json",
        "resources/effects/sources/vortex_force.py",
        "resources/examples/05_custom_effects_vortex.blend",
        "resources/examples/06_custom_effects_stack.blend",
        "resources/examples/07_warp_code_authoring.blend",
        "resources/examples/custom_effects/renders/05_custom_effects_vortex.png",
        "resources/examples/custom_effects/renders/06_custom_effects_stack.png",
        "resources/examples/custom_effects/renders/07_warp_code_authoring.png",
    )
    missing_required = [name for name in required if not (root / name).is_file()]
    if missing_required:
        custom_missing = [
            item for item in missing_required
            if item.startswith(("sim/effects/", "blender/effect_", "resources/effects/"))
            or "custom_effects" in item
        ]
        if custom_missing:
            raise RuntimeError(f"missing required custom effect release files: {custom_missing}")
        raise RuntimeError(f"missing required release files: {missing_required}")

    manifest = _read_toml(root / "blender_manifest.toml")
    lock = _read_json(_tool_root_for(root) / "wheels.lock.json")
    collision_guard = _collision_guard(root)
    files = _locked_files(lock)
    if lock.get("warp_version") != REQUIRED_WARP_VERSION:
        raise RuntimeError(f"Warp version lock must be {REQUIRED_WARP_VERSION}")
    if lock.get("distribution_version") != REQUIRED_WARP_DISTRIBUTION:
        raise RuntimeError(f"Warp distribution lock must be {REQUIRED_WARP_DISTRIBUTION}")
    if _assigned_literal(root / "version.py", "CERTIFIED_WARP_VERSION") != REQUIRED_WARP_VERSION:
        raise RuntimeError("version.py CERTIFIED_WARP_VERSION does not match the Warp lock")
    if (
        _assigned_literal(root / "version.py", "CERTIFIED_WARP_DISTRIBUTION")
        != REQUIRED_WARP_DISTRIBUTION
    ):
        raise RuntimeError("version.py CERTIFIED_WARP_DISTRIBUTION does not match the Warp lock")

    version = _assigned_literal(root / "version.py", "VERSION")
    extension_id = _assigned_literal(root / "version.py", "EXTENSION_ID")
    if not isinstance(version, tuple) or not version or not all(isinstance(item, int) for item in version):
        raise RuntimeError("VERSION must be a non-empty integer tuple")
    version_string = ".".join(str(item) for item in version)

    if extension_id != REQUIRED_EXTENSION_ID or manifest.get("id") != extension_id:
        raise RuntimeError("unexpected extension id")
    if manifest.get("version") != version_string:
        raise RuntimeError("manifest version does not match version.py")
    if manifest.get("license") != REQUIRED_LICENSE:
        raise RuntimeError("unexpected extension license")
    if manifest.get("blender_version_min") != REQUIRED_BLENDER_MIN:
        raise RuntimeError(f"Blender minimum version must be {REQUIRED_BLENDER_MIN}")

    platforms = tuple(manifest.get("platforms", ()))
    if platforms != REQUIRED_PLATFORMS:
        raise RuntimeError(
            f"supported platforms must be {list(REQUIRED_PLATFORMS)}, got {list(platforms)}"
        )
    lock_platforms = {str(item.get("platform", "")) for item in files.values()}
    if not lock_platforms or not lock_platforms.issubset(platforms):
        raise RuntimeError("locked wheel platforms must be a supported platform subset")

    manifest_wheels = manifest.get("wheels")
    expected_wheels = sorted(f"./{BUNDLED_WHEEL_DIR}/{name}" for name in files)
    if not isinstance(manifest_wheels, list) or sorted(manifest_wheels) != expected_wheels:
        raise RuntimeError(
            f"manifest wheels must be exactly the locked Warp wheels {expected_wheels}, "
            f"got {manifest_wheels!r}"
        )
    if "generated" in manifest.get("build", {}):
        raise RuntimeError("tracked source manifest must not contain generated metadata")

    build_patterns = set(manifest.get("build", {}).get("paths_exclude_pattern", ()))
    required_patterns = {
        "__pycache__/", ".*", "*.zip", "*.whl", "wheels/",
        "tests/", "demo_outputs/", "artifacts/",
        "pyproject.toml", "requirements-dev.txt",
    }
    if not required_patterns.issubset(build_patterns):
        missing_patterns = sorted(required_patterns - build_patterns)
        raise RuntimeError(f"manifest build exclusions are incomplete: {missing_patterns}")

    init_source = (root / "__init__.py").read_text(encoding="utf-8")
    if "bl_info" in init_source:
        raise RuntimeError("legacy bl_info metadata is not allowed in an extension")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for marker in ("NVIDIA Warp", "Apache License, Version 2.0", "libmathdx"):
        if marker not in notices:
            raise RuntimeError(f"THIRD_PARTY_NOTICES.md is missing {marker!r}")
    if not (root / "LICENSE").read_text(encoding="utf-8").strip():
        raise RuntimeError("LICENSE is empty")

    render_asset = (
        root / "resources" / "geometry_nodes" / "fluidsim_render_nodes.blend"
    )
    if not render_asset.is_file() or render_asset.stat().st_size <= 50_000:
        raise RuntimeError("bundled Geometry Nodes asset is missing or unexpectedly small")
    if not _looks_like_blender_file(render_asset.read_bytes()[:7]):
        raise RuntimeError("bundled Geometry Nodes asset is not a Blender file")
    if list(render_asset.parent.glob("*.blend1")):
        raise RuntimeError("Blender backup files must not be shipped as render assets")

    forbidden_source_wheels = []
    ignored_roots = {".git", "build", "dist", "artifacts"}
    allowed_wheels = {f"{BUNDLED_WHEEL_DIR}/{name}" for name in files}
    for path in root.rglob("*.whl"):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in ignored_roots:
            continue
        if rel.as_posix() in allowed_wheels:
            continue
        forbidden_source_wheels.append(str(rel))
    if forbidden_source_wheels:
        raise RuntimeError(
            "only the locked Warp wheels may be stored in the deliverable source tree: "
            f"{sorted(forbidden_source_wheels)}"
        )

    _verify_source_collision_guard(root, collision_guard)

    for filename, metadata in sorted(files.items()):
        expected_sha = str(metadata.get("sha256", ""))
        expected_size = metadata.get("size")
        platform = str(metadata.get("platform", ""))
        source_name = str(metadata.get("source", ""))
        url = str(metadata.get("url", ""))
        if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
            raise RuntimeError(f"invalid SHA-256 lock for {filename}")
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise RuntimeError(f"invalid size lock for {filename}")
        if platform not in REQUIRED_PLATFORMS:
            raise RuntimeError(f"invalid platform lock for {filename}: {platform}")
        if not source_name:
            raise RuntimeError(f"wheel provenance is missing: {filename}")
        if url and not url.startswith("https://"):
            raise RuntimeError(f"wheel URL must use HTTPS when present: {filename}")

    return SourceVerification(
        extension_id=str(extension_id),
        version=version_string,
        platforms=platforms,
        bundled_warp_wheels=tuple(sorted(files)),
        wheel_platforms=tuple(
            (name, str(files[name].get("platform", ""))) for name in sorted(files)
        ),
        wheel_sha256=tuple(
            (name, str(files[name].get("sha256", ""))) for name in sorted(files)
        ),
    )

def _archive_relative_names(archive: zipfile.ZipFile) -> dict[str, str]:
    names = [name for name in archive.namelist() if name and not name.endswith("/")]
    if not names:
        raise RuntimeError("extension package is empty")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {name}")
    manifests = [name for name in names if PurePosixPath(name).name == "blender_manifest.toml"]
    if len(manifests) != 1:
        raise RuntimeError(f"expected exactly one blender_manifest.toml, found {manifests}")
    prefix = str(PurePosixPath(manifests[0]).parent)
    prefix = "" if prefix == "." else prefix.rstrip("/") + "/"
    relative: dict[str, str] = {}
    for name in names:
        if prefix and not name.startswith(prefix):
            raise RuntimeError("archive contains files outside the extension root")
        normalized = name[len(prefix) :] if prefix else name
        relative[normalized] = name
    return relative


def _platform_from_package_name(path: Path, extension_id: str, version: str) -> str:
    matches = [
        platform
        for platform in REQUIRED_PLATFORMS
        if path.name == f"{extension_id}-{version}-{platform.replace('-', '_')}.zip"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"unexpected split package filename: {path.name}")
    return matches[0]


def _is_forbidden_runtime_payload(relative: str) -> bool:
    path = PurePosixPath(relative)
    lower_parts = tuple(part.lower() for part in path.parts)
    filename = path.name.lower()
    if path.suffix.lower() == ".whl":
        return True
    if lower_parts and lower_parts[0] == "wheels":
        return True
    if any(part == "warp" or part.startswith("warp_lang-") for part in lower_parts):
        return True
    if filename.startswith("blender-") and filename.endswith(
        (".zip", ".tar.xz", ".tar.gz", ".tgz")
    ):
        return True
    return filename == "blender" and (not lower_parts or lower_parts[0] != "blender")


def _validate_archive_members(
    names: Mapping[str, str], allowed_payloads: frozenset[str] = frozenset()
) -> None:
    for relative in names:
        path = PurePosixPath(relative)
        if relative in allowed_payloads:
            continue
        if _is_forbidden_runtime_payload(relative):
            raise RuntimeError(f"forbidden bundled runtime payload: {relative}")
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            raise RuntimeError(f"compiled Python artifact leaked into package: {relative}")
        if path.parts and path.parts[0] in RELEASE_ONLY_PATHS:
            raise RuntimeError(f"release-only path leaked into package: {relative}")
        if relative in RELEASE_ONLY_FILES or relative in RELEASE_ONLY_RELATIVE_PATHS:
            raise RuntimeError(f"release-only file leaked into package: {relative}")


def _validate_required_package_content(
    package: Path,
    archive: zipfile.ZipFile,
    names: Mapping[str, str],
    collision_guard: Mapping[str, str | None],
) -> None:
    required = {
        "blender_manifest.toml",
        "__init__.py",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "release/__init__.py",
        "release/run_headless_bake.py",
        "blender/setup_wizard.py",
        "blender/geometry_nodes.py",
        "sim/attributes.py",
        "sim/surface_tension.py",
        "sim/surface_tension_math.py",
        "presets/schema.py",
        "resources/presets/preview_water.json",
        "resources/presets/water_surface_tension.json",
        "resources/presets/stylized_high_tension.json",
        "resources/geometry_nodes/fluidsim_render_nodes.blend",
        "sim/effects/schema.py",
        "sim/effects/runtime.py",
        "sim/effects/package.py",
        "sim/effects/templates.py",
        "blender/effects.py",
        "blender/effect_ui.py",
        "blender/effect_workspace.py",
        "resources/effects/vortex_force.json",
        "resources/effects/sources/vortex_force.py",
        "resources/examples/05_custom_effects_vortex.blend",
        "resources/examples/06_custom_effects_stack.blend",
        "resources/examples/07_warp_code_authoring.blend",
        "resources/examples/custom_effects/renders/05_custom_effects_vortex.png",
        "resources/examples/custom_effects/renders/06_custom_effects_stack.png",
        "resources/examples/custom_effects/renders/07_warp_code_authoring.png",
    }
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"package {package.name} is missing files: {missing}")
    for relative, stale_sha256 in collision_guard.items():
        archive_name = names.get(relative)
        if stale_sha256 is None:
            if archive_name is not None:
                raise RuntimeError(
                    f"package {package.name} contains retired integration owner: {relative}"
                )
            continue
        if archive_name is None:
            raise RuntimeError(
                f"package {package.name} is missing protected integration owner: {relative}"
            )
        if _sha256_bytes(archive.read(archive_name)) == stale_sha256:
            raise RuntimeError(
                "stale candidate implementation replaced adjudicated owner in "
                f"{package.name}: {relative}"
            )
    if any(PurePosixPath(name).suffix == ".blend1" for name in names):
        raise RuntimeError(f"Blender backup asset leaked into {package.name}")
    effect_json = [
        name for name in names
        if name.startswith("resources/effects/")
        and "/sources/" not in name and name.endswith(".json")
    ]
    effect_sources = [
        name for name in names
        if name.startswith("resources/effects/sources/") and name.endswith(".py")
    ]
    if len(effect_json) != 8 or len(effect_sources) != 8:
        raise RuntimeError(
            f"package {package.name} must contain eight built-in effect definitions "
            "and eight source files"
        )
    asset_payload = archive.read(
        names["resources/geometry_nodes/fluidsim_render_nodes.blend"]
    )
    if len(asset_payload) <= 50_000 or not _looks_like_blender_file(
            asset_payload[:7]):
        raise RuntimeError(f"invalid Geometry Nodes asset in {package.name}")


def _validate_embedded_manifest_and_payload(
    package: Path,
    platform: str,
    archive: zipfile.ZipFile,
    names: Mapping[str, str],
    source: SourceVerification,
    reference_payloads: dict[str, str] | None,
    reference_manifest: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    embedded = tomllib.loads(archive.read(names["blender_manifest.toml"]).decode("utf-8"))
    if embedded.get("id") != source.extension_id or embedded.get("version") != source.version:
        raise RuntimeError(f"embedded manifest identity mismatch: {package.name}")
    if embedded.get("license") != REQUIRED_LICENSE:
        raise RuntimeError(f"embedded manifest license mismatch: {package.name}")
    if tuple(embedded.get("platforms", ())) != source.platforms:
        raise RuntimeError(
            f"embedded source platform mismatch: expected {list(source.platforms)}, "
            f"got {embedded.get('platforms')}"
        )
    expected_wheels = sorted(
        f"./{BUNDLED_WHEEL_DIR}/{name}" for name in source.bundled_warp_wheels
    )
    if sorted(embedded.get("wheels") or []) != expected_wheels:
        raise RuntimeError(f"embedded manifest wheels mismatch: {package.name}")
    generated = embedded.get("build", {}).get("generated", {})
    if generated.get("platforms") != [platform]:
        raise RuntimeError(f"generated platform metadata mismatch: {package.name}")
    platform_wheel = f"./{BUNDLED_WHEEL_DIR}/{source.wheel_for_platform(platform)}"
    if generated.get("wheels", []) != [platform_wheel]:
        raise RuntimeError(
            f"generated wheels must be [{platform_wheel!r}]: {package.name}"
        )

    normalized_manifest = dict(embedded)
    normalized_build = dict(normalized_manifest.get("build", {}))
    normalized_build.pop("generated", None)
    normalized_manifest["build"] = normalized_build
    # Each split package carries only its own platform's Warp wheel; compare the
    # rest of the payload across platforms.
    payloads = {
        relative: _sha256_bytes(archive.read(archive_name))
        for relative, archive_name in names.items()
        if relative != "blender_manifest.toml"
        and PurePosixPath(relative).parts[0] != BUNDLED_WHEEL_DIR
    }
    if reference_payloads is None:
        return payloads, normalized_manifest
    if payloads != reference_payloads or normalized_manifest != reference_manifest:
        raise RuntimeError(
            "split-platform payloads differ outside generated platform metadata"
        )
    return payloads, normalized_manifest


def _validate_bundled_wheel(
    package: Path,
    platform: str,
    archive: zipfile.ZipFile,
    names: Mapping[str, str],
    source: SourceVerification,
) -> str:
    """Require exactly the platform's locked Warp wheel, byte-identical to the lock."""
    wheel_name = source.wheel_for_platform(platform)
    relative = f"{BUNDLED_WHEEL_DIR}/{wheel_name}"
    bundled = sorted(
        name for name in names
        if PurePosixPath(name).parts[0] == BUNDLED_WHEEL_DIR
    )
    if bundled != [relative]:
        raise RuntimeError(
            f"package {package.name} must bundle exactly {relative}, found {bundled}"
        )
    expected_sha = dict(source.wheel_sha256)[wheel_name]
    if _sha256_bytes(archive.read(names[relative])) != expected_sha:
        raise RuntimeError(f"bundled Warp wheel checksum mismatch in {package.name}")
    return relative


def inspect_packages(root: Path, packages: Iterable[Path]) -> list[ArtifactRecord]:
    """Inspect split archives: exactly one locked Warp wheel, no other binaries."""
    root = Path(root).resolve()
    source = verify_source_tree(root)
    collision_guard = _collision_guard(root)
    package_paths = sorted((Path(path).resolve() for path in packages), key=lambda path: path.name)
    if not package_paths:
        raise RuntimeError("no split-platform packages were provided for inspection")
    allowed_names = {
        f"{source.extension_id}-{source.version}-{platform.replace('-', '_')}.zip"
        for platform in source.platforms
    }
    unexpected_names = sorted(path.name for path in package_paths if path.name not in allowed_names)
    if unexpected_names:
        raise RuntimeError(f"unexpected split package filenames: {unexpected_names}")
    if len({path.name for path in package_paths}) != len(package_paths):
        raise RuntimeError("duplicate split-platform package paths were provided")

    records: list[ArtifactRecord] = []
    reference_payloads: dict[str, str] | None = None
    reference_manifest: dict[str, Any] | None = None
    for package in package_paths:
        if not package.is_file():
            raise RuntimeError(f"missing built package: {package}")
        platform = _platform_from_package_name(package, source.extension_id, source.version)
        with zipfile.ZipFile(package) as archive:
            names = _archive_relative_names(archive)
            wheel = _validate_bundled_wheel(package, platform, archive, names, source)
            _validate_archive_members(names, frozenset({wheel}))
            _validate_required_package_content(package, archive, names, collision_guard)
            reference_payloads, reference_manifest = _validate_embedded_manifest_and_payload(
                package,
                platform,
                archive,
                names,
                source,
                reference_payloads,
                reference_manifest,
            )

        records.append(
            ArtifactRecord(
                path=package,
                platform=platform,
                runtime_dependency="bundled",
                size=package.stat().st_size,
                sha256=_sha256_path(package),
            )
        )
    return sorted(records, key=lambda item: item.platform)

def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after an atomic metadata replacement."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably replace a small release metadata file without exposing partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_checksums(output: Path, records: Sequence[ArtifactRecord]) -> Path:
    path = output / "SHA256SUMS.txt"
    text = "".join(f"{record.sha256}  {record.path.name}\n" for record in records)
    _write_bytes_atomic(path, text.encode("utf-8"))
    return path


def _fetch_locked_wheels(root: Path) -> None:
    """Place every locked Warp wheel in the add-on's wheels/ directory."""
    sys.path.insert(0, str(_tool_root_for(root)))
    try:
        from fetch_warp_wheels import fetch_wheels
    finally:
        sys.path.pop(0)
    fetch_wheels(
        lock_path=_tool_root_for(root) / "wheels.lock.json",
        wheel_dir=root / BUNDLED_WHEEL_DIR,
    )


def build(
    root: Path,
    blender: Path,
    output: Path,
    *,
    runner: Runner = _run,
    run_python_checks: bool = True,
    run_lint: bool = True,
    fetcher: Callable[[Path], object] | None = None,
) -> list[Path]:
    """Fetch the locked wheels, then run static gates, Blender validation,
    split build, and archive inspection."""
    root = Path(root).resolve()
    blender = Path(blender)
    output = Path(output).resolve()
    (fetcher or _fetch_locked_wheels)(root)
    source = verify_source_tree(root)
    output.mkdir(parents=True, exist_ok=True)
    for path in output.glob(f"{source.extension_id}-{source.version}-*.zip"):
        path.unlink()
    checksum_path = output / "SHA256SUMS.txt"
    if checksum_path.exists():
        checksum_path.unlink()

    if run_python_checks:
        if run_lint:
            runner([sys.executable, "-m", "ruff", "check", "."], root)
        runner([sys.executable, "-m", "pytest", "-q"], root)
        runner([sys.executable, "-m", "compileall", "-q", "-f", "."], root)
    blender_package_prefix = [
        str(blender),
        "--factory-startup",
        "--background",
        "--offline-mode",
    ]
    runner(
        [*blender_package_prefix, "--command", "extension", "validate", str(root)],
        root,
    )
    runner(
        [
            *blender_package_prefix,
            "--command",
            "extension",
            "build",
            "--split-platforms",
            "--source-dir",
            str(root),
            "--output-dir",
            str(output),
            "--verbose",
        ],
        root,
    )
    packages = sorted(output.glob(f"{source.extension_id}-{source.version}-*.zip"))
    expected_names = {
        f"{source.extension_id}-{source.version}-{platform.replace('-', '_')}.zip"
        for platform in source.platforms
    }
    actual_names = {path.name for path in packages}
    if actual_names != expected_names:
        raise RuntimeError(
            f"expected split-platform packages {sorted(expected_names)}, "
            f"found {sorted(actual_names)}"
        )
    records = inspect_packages(root, packages)
    _write_checksums(output, records)
    return [record.path for record in records]


def run_installed_script(
    blender: Path,
    package: Path,
    repo_dir: Path,
    script: Path,
    *,
    repeats: int = 1,
    runner: Runner = _run,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    """Install one archive into an isolated repository and run a Blender script."""
    blender = Path(blender)
    package = Path(package).resolve()
    repo_dir = Path(repo_dir).resolve()
    script = Path(script).resolve()
    if not package.is_file():
        raise RuntimeError(f"package does not exist: {package}")
    if not script.is_file():
        raise RuntimeError(f"Blender script does not exist: {script}")
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    profile_dir = repo_dir.parent / f"{repo_dir.name}-profile"
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)
    env = _blender_process_environment(os.environ)
    isolated_paths = {
        "BLENDER_USER_CONFIG": profile_dir / "config",
        "BLENDER_USER_SCRIPTS": profile_dir / "scripts",
        "BLENDER_USER_EXTENSIONS": profile_dir / "extensions",
        "BLENDER_USER_DATAFILES": profile_dir / "datafiles",
        "WARP_CACHE_PATH": profile_dir / "warp-cache",
        "TEMP": profile_dir / "temp",
        "TMP": profile_dir / "temp",
    }
    for path in isolated_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env.update({key: str(path) for key, path in isolated_paths.items()})
    env["FLUIDSIM_CI_REPOSITORY"] = str(repo_dir)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})

    cwd = script.parent
    runner(
        [
            str(blender),
            "--command",
            "extension",
            "repo-add",
            "fluidsim_ci",
            "--name",
            "FluidSim CI",
            "--directory",
            str(repo_dir),
            "--clear-all",
        ],
        cwd,
        env=env,
    )
    runner(
        [
            str(blender),
            "--command",
            "extension",
            "install-file",
            "-r",
            "fluidsim_ci",
            "-e",
            str(package),
        ],
        cwd,
        env=env,
    )
    for _ in range(repeats):
        runner(
            [
                str(blender),
                "--background",
                "--offline-mode",
                "--python-exit-code",
                "1",
                "--python",
                str(script),
            ],
            cwd,
            env=env,
        )


def run_installed_smoke(
    blender: Path,
    package: Path,
    repo_dir: Path,
    script: Path,
    *,
    runner: Runner = _run,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    """Run registration smoke coverage twice, then uninstall from the clean profile."""
    run_installed_script(
        blender,
        package,
        repo_dir,
        script,
        repeats=2,
        runner=runner,
        extra_env=extra_env,
    )

    report_value = str((extra_env or {}).get("FLUIDSIM_SMOKE_REPORT", ""))
    if report_value and runner is _run:
        report_path = Path(report_value)
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read smoke report {report_path}: {exc}") from exc
        if report.get("status") != "passed" or report.get("run_count") != 2:
            raise RuntimeError(f"installed smoke did not record two passing runs: {report}")

    profile_dir = Path(repo_dir).resolve().parent / f"{Path(repo_dir).name}-profile"
    env = _blender_process_environment(os.environ)
    isolated_paths = {
        "BLENDER_USER_CONFIG": profile_dir / "config",
        "BLENDER_USER_SCRIPTS": profile_dir / "scripts",
        "BLENDER_USER_EXTENSIONS": profile_dir / "extensions",
        "BLENDER_USER_DATAFILES": profile_dir / "datafiles",
        "WARP_CACHE_PATH": profile_dir / "warp-cache",
        "TEMP": profile_dir / "temp",
        "TMP": profile_dir / "temp",
    }
    env.update({key: str(path) for key, path in isolated_paths.items()})
    env["FLUIDSIM_CI_REPOSITORY"] = str(Path(repo_dir).resolve())
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    runner(
        [
            str(blender),
            "--command",
            "extension",
            "remove",
            "fluidsim_ci.fluidsim_warp",
        ],
        Path(script).resolve().parent,
        env=env,
    )
    installed_root = Path(repo_dir).resolve() / "fluidsim_warp"
    if installed_root.exists():
        raise RuntimeError(f"extension uninstall left package files behind: {installed_root}")


def _record_payload(records: Sequence[ArtifactRecord]) -> list[dict[str, object]]:
    return [record.as_dict() for record in records]


def _package_for_platform(packages: Sequence[Path], platform: str) -> Path:
    suffix = f"-{platform.replace('-', '_')}.zip"
    matches = [path for path in packages if path.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {platform} package, found {matches}")
    return matches[0]


def _write_json_report(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-python-checks", action="store_true")
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--smoke-platform", choices=REQUIRED_PLATFORMS)
    parser.add_argument("--gpu-regression-platform", choices=REQUIRED_PLATFORMS)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    blender = args.blender.resolve() if args.blender is not None else None
    repository = args.repository.resolve() if args.repository is not None else None
    json_report = args.json_report.resolve() if args.json_report is not None else None

    try:
        if args.verify_only:
            result = verify_source_tree(root)
            payload: dict[str, object] = {
                "mode": "verify-only",
                "source": asdict(result),
                "release_ready": result.release_ready,
            }
            if json_report:
                _write_json_report(json_report, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if result.release_ready else 2

        if blender is None:
            parser.error("--blender is required unless --verify-only is used")
        packages = build(
            root,
            blender,
            output,
            run_python_checks=not args.skip_python_checks,
            run_lint=not args.skip_lint,
        )
        records = inspect_packages(root, packages)
        payload = {
            "mode": "build",
            "artifacts": _record_payload(records),
            "release_ready": True,
        }
        repo_dir = repository or output / "fluidsim-ci-repository"
        if args.smoke_platform:
            package = _package_for_platform(packages, args.smoke_platform)
            smoke_report = output / "smoke-test-report.json"
            if smoke_report.exists():
                smoke_report.unlink()
            run_installed_smoke(
                blender,
                package,
                repo_dir,
                TOOL_ROOT / "smoke_test_blender.py",
                extra_env={"FLUIDSIM_SMOKE_REPORT": str(smoke_report)},
            )
            payload["smoke_platform"] = args.smoke_platform
            payload["smoke_report"] = str(smoke_report)
        if args.gpu_regression_platform:
            package = _package_for_platform(packages, args.gpu_regression_platform)
            report_path = output / "gpu-regression-report.json"
            if report_path.exists():
                report_path.unlink()
            run_installed_script(
                blender,
                package,
                repo_dir,
                TOOL_ROOT / "gpu_regression_blender.py",
                extra_env={"FLUIDSIM_GPU_REPORT": str(report_path)},
            )
            payload["gpu_regression_platform"] = args.gpu_regression_platform
            payload["gpu_report"] = str(report_path)
        if json_report:
            _write_json_report(json_report, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        payload = {"release_ready": False, "error": f"{type(exc).__name__}: {exc}"}
        if json_report:
            _write_json_report(json_report, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
