"""Blender-facing built-in and user Effect Package library services."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..sim.effects.builtins import builtin_effect_definitions
from ..sim.effects.library import (
    EffectLibrary,
    EffectUpdateComparison,
    LibraryDiagnostic,
    compare_effect_versions,
    merge_effect_definitions,
    renamed_definition,
    unique_copy_definition,
)
from ..sim.effects.package import read_effect_package, write_effect_package
from ..sim.effects.schema import EffectDefinitionV2


@dataclass(frozen=True)
class InstalledEffectPackage:
    path: Path
    definition: EffectDefinitionV2


@dataclass(frozen=True)
class EffectLibrarySnapshot:
    library: EffectLibrary
    diagnostics: tuple[LibraryDiagnostic, ...]
    installed: tuple[InstalledEffectPackage, ...]


def builtin_library() -> EffectLibrary:
    return EffectLibrary(builtin_effect_definitions())


def _effect_user_root(*, create: bool) -> Path:
    from .effects import effect_user_root

    return effect_user_root(create=create)


def user_library_path() -> Path:
    path = _effect_user_root(create=True) / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def installed_user_entries() -> tuple[tuple[InstalledEffectPackage, ...], tuple[LibraryDiagnostic, ...]]:
    entries: list[InstalledEffectPackage] = []
    diagnostics: list[LibraryDiagnostic] = []
    for path in sorted(user_library_path().glob("*.fseffect")):
        try:
            definition = read_effect_package(path)
        except ValueError as exc:
            diagnostics.append(LibraryDiagnostic(
                source=str(path),
                code="INVALID_PACKAGE",
                message=str(exc),
            ))
        else:
            entries.append(InstalledEffectPackage(path, definition))
    return tuple(entries), tuple(diagnostics)


def installed_user_definitions():
    """Compatibility result used by older callers: definitions plus path/message pairs."""

    entries, diagnostics = installed_user_entries()
    return (
        tuple(item.definition for item in entries),
        tuple((Path(item.source), item.message) for item in diagnostics),
    )


def library_snapshot() -> EffectLibrarySnapshot:
    installed, read_diagnostics = installed_user_entries()
    merged = merge_effect_definitions(
        builtin_effect_definitions(),
        tuple((str(item.path), item.definition) for item in installed),
    )
    return EffectLibrarySnapshot(
        library=merged.library,
        diagnostics=tuple((*read_diagnostics, *merged.diagnostics)),
        installed=installed,
    )


def combined_library() -> EffectLibrary:
    return library_snapshot().library


def library_diagnostics() -> tuple[LibraryDiagnostic, ...]:
    return library_snapshot().diagnostics


def install_package(path: str | Path, *, conflict: str = "cancel"):
    definition = replace(read_effect_package(path), built_in=False)
    if conflict not in {"cancel", "keep_both", "replace"}:
        raise ValueError(f"unknown conflict policy {conflict!r}")

    snapshot = library_snapshot()
    builtin_ids = {item.effect_id for item in builtin_effect_definitions()}
    installed_matches = tuple(
        item for item in snapshot.installed if item.definition.effect_id == definition.effect_id
    )
    existing_ids = {
        item.effect_id for item in snapshot.library.definitions()
    } | {item.definition.effect_id for item in snapshot.installed}
    identity_conflict = definition.effect_id in existing_ids

    if identity_conflict:
        if conflict == "cancel":
            raise FileExistsError(
                f"effect ID {definition.effect_id!r} is already installed"
            )
        if conflict == "keep_both":
            definition = unique_copy_definition(definition, existing_ids)
            installed_matches = ()
        elif definition.effect_id in builtin_ids:
            raise ValueError(
                "cannot replace a built-in effect identity; import with Keep Both instead"
            )

    destination = user_library_path() / f"{definition.effect_id}-{definition.version}.fseffect"
    if destination.exists() and not identity_conflict:
        if conflict == "cancel":
            raise FileExistsError(destination)
        if conflict == "keep_both":
            definition = unique_copy_definition(definition, existing_ids)
            destination = user_library_path() / (
                f"{definition.effect_id}-{definition.version}.fseffect"
            )

    # Re-serialize after validation; never copy malformed or metadata-spoofed bytes.
    # write_effect_package uses fsync plus os.replace, so an exact-version update is atomic.
    report = write_effect_package(destination, definition)
    if conflict == "replace":
        for item in installed_matches:
            if item.path.resolve() != report.path.resolve():
                item.path.unlink(missing_ok=True)
    return report


def export_definition(path: str | Path, definition):
    return write_effect_package(path, definition)


def delete_package(path: str | Path) -> None:
    candidate = Path(path).resolve()
    root = user_library_path().resolve()
    if root not in candidate.parents or candidate.suffix != ".fseffect":
        raise ValueError("can delete only an installed .fseffect package")
    candidate.unlink()


def delete_definition(effect_id: str) -> tuple[Path, ...]:
    removed = []
    installed, _diagnostics = installed_user_entries()
    for item in installed:
        if item.definition.effect_id == str(effect_id):
            delete_package(item.path)
            removed.append(item.path)
    if not removed:
        raise FileNotFoundError(f"no installed package provides effect ID {effect_id!r}")
    return tuple(removed)


def reveal_path() -> Path:
    return user_library_path()


def installed_package_for_effect(effect_id: str) -> InstalledEffectPackage:
    matches = tuple(
        item for item in installed_user_entries()[0]
        if item.definition.effect_id == str(effect_id)
    )
    if not matches:
        raise FileNotFoundError(f"no installed package provides effect ID {effect_id!r}")
    if len(matches) > 1:
        raise ValueError(
            f"multiple installed packages provide effect ID {effect_id!r}; "
            "resolve the library conflict before continuing"
        )
    return matches[0]


def compare_package_update(path: str | Path) -> EffectUpdateComparison:
    candidate = read_effect_package(path)
    builtin_ids = {item.effect_id for item in builtin_effect_definitions()}
    if candidate.effect_id in builtin_ids:
        return EffectUpdateComparison(
            effect_id=candidate.effect_id,
            candidate_version=candidate.version,
            status="BUILTIN_CONFLICT",
            message=(
                f"{candidate.effect_id!r} is a built-in identity and cannot be updated "
                "from a user package. Import with Keep Both instead."
            ),
        )

    installed, _diagnostics = installed_user_entries()
    matches = tuple(
        item for item in installed
        if item.definition.effect_id == candidate.effect_id
    )
    if not matches:
        return EffectUpdateComparison(
            effect_id=candidate.effect_id,
            candidate_version=candidate.version,
            status="NOT_INSTALLED",
            message=f"{candidate.name} is not installed in the user effect library.",
        )
    if len(matches) > 1:
        return EffectUpdateComparison(
            effect_id=candidate.effect_id,
            candidate_version=candidate.version,
            status="AMBIGUOUS",
            message=(
                f"Multiple installed packages provide {candidate.effect_id!r}; "
                "resolve the identity conflict before comparing updates."
            ),
        )

    current = matches[0]
    status = compare_effect_versions(current.definition, candidate)
    messages = {
        "IDENTICAL": "Candidate content and version are identical to the installed package.",
        "MODIFIED": (
            "Candidate uses the same semantic version core but its validated definition "
            "differs from the installed package."
        ),
        "NEWER": "Candidate is newer than the installed package.",
        "OLDER": "Candidate is older than the installed package.",
    }
    return EffectUpdateComparison(
        effect_id=candidate.effect_id,
        candidate_version=candidate.version,
        installed_version=current.definition.version,
        status=status,
        message=messages[status],
        installed_source=str(current.path),
    )


def rename_installed_package(
    path: str | Path,
    *,
    new_effect_id: str,
    new_name: str,
) -> InstalledEffectPackage:
    candidate = Path(path).resolve()
    root = user_library_path().resolve()
    if root not in candidate.parents or candidate.suffix.lower() != ".fseffect":
        raise ValueError("can rename only an installed .fseffect package")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    definition = read_effect_package(candidate)
    renamed = renamed_definition(definition, new_effect_id, new_name)
    builtin_ids = {item.effect_id for item in builtin_effect_definitions()}
    if renamed.effect_id in builtin_ids:
        raise ValueError(
            f"cannot rename a user package to built-in effect ID {renamed.effect_id!r}"
        )

    installed, _diagnostics = installed_user_entries()
    for item in installed:
        if item.path.resolve() == candidate:
            continue
        if item.definition.effect_id == renamed.effect_id:
            raise FileExistsError(
                f"effect ID {renamed.effect_id!r} is already installed by {item.path.name}"
            )

    destination = root / f"{renamed.effect_id}-{renamed.version}.fseffect"
    if destination.exists() and destination.resolve() != candidate:
        raise FileExistsError(f"destination package already exists: {destination.name}")

    report = write_effect_package(destination, renamed)
    if candidate != report.path.resolve():
        candidate.unlink()
    return InstalledEffectPackage(report.path.resolve(), renamed)
