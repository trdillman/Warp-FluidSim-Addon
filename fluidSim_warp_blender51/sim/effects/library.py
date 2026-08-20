"""Search and compatibility services for Custom Effects."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .schema import EffectDefinitionV2


@dataclass(frozen=True)
class LibrarySearchResult:
    definition: EffectDefinitionV2
    score: int


@dataclass(frozen=True)
class LibraryDiagnostic:
    source: str
    code: str
    message: str
    effect_id: str = ""


@dataclass(frozen=True)
class LibraryMergeResult:
    library: "EffectLibrary"
    diagnostics: tuple[LibraryDiagnostic, ...]


@dataclass(frozen=True)
class EffectUpdateComparison:
    effect_id: str
    candidate_version: str
    installed_version: str = ""
    status: str = ""
    message: str = ""
    installed_source: str = ""


def _semantic_version_core(version: str) -> tuple[int, int, int]:
    core = str(version).split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid semantic version {version!r}")
    return tuple(int(part) for part in parts)


def compare_effect_versions(
    installed: EffectDefinitionV2,
    candidate: EffectDefinitionV2,
) -> str:
    """Classify a same-identity candidate against one installed definition."""

    if installed.effect_id != candidate.effect_id:
        raise ValueError("effect update comparison requires the same effect ID")
    if installed == candidate:
        return "IDENTICAL"
    if installed.version == candidate.version:
        return "MODIFIED"
    installed_core = _semantic_version_core(installed.version)
    candidate_core = _semantic_version_core(candidate.version)
    if candidate_core > installed_core:
        return "NEWER"
    if candidate_core < installed_core:
        return "OLDER"
    return "MODIFIED"


def renamed_definition(
    definition: EffectDefinitionV2,
    new_effect_id: str,
    new_name: str,
) -> EffectDefinitionV2:
    """Create a local package definition with a new public identity."""

    effect_id = str(new_effect_id).strip()
    name = str(new_name).strip()
    if not effect_id:
        raise ValueError("new effect ID cannot be empty")
    if not name:
        raise ValueError("new effect name cannot be empty")
    return replace(
        definition,
        effect_id=effect_id,
        name=name,
        built_in=False,
        package_origin="local-rename",
    )


class EffectLibrary:
    def __init__(self, definitions=()):
        items = tuple(definitions)
        by_id = {item.effect_id: item for item in items}
        if len(by_id) != len(items):
            raise ValueError("effect library contains duplicate effect IDs")
        self._definitions = tuple(sorted(items, key=lambda item: (item.category, item.name)))
        self._by_id = by_id

    def definitions(self) -> tuple[EffectDefinitionV2, ...]:
        return self._definitions

    def get(self, effect_id: str) -> EffectDefinitionV2:
        try:
            return self._by_id[str(effect_id)]
        except KeyError as exc:
            raise KeyError(f"unknown effect {effect_id!r}") from exc

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self._definitions}))

    def search(self, query: str = "", *, category: str = "") -> tuple[EffectDefinitionV2, ...]:
        words = tuple(word for word in str(query).lower().split() if word)
        matches = []
        for definition in self._definitions:
            if category and definition.category != category:
                continue
            haystack = " ".join((
                definition.effect_id, definition.name, definition.summary,
                definition.description, definition.category, *definition.tags,
            )).lower()
            if all(word in haystack for word in words):
                matches.append(definition)
        return tuple(matches)


def merge_effect_definitions(
    builtins: tuple[EffectDefinitionV2, ...] | list[EffectDefinitionV2],
    user_items: tuple[tuple[str, EffectDefinitionV2], ...] | list[tuple[str, EffectDefinitionV2]],
) -> LibraryMergeResult:
    """Merge installed definitions without allowing identity shadowing."""

    accepted = list(builtins)
    occupied = {item.effect_id: "built-in" for item in accepted}
    diagnostics: list[LibraryDiagnostic] = []
    for source, definition in user_items:
        previous = occupied.get(definition.effect_id)
        if previous is not None:
            diagnostics.append(LibraryDiagnostic(
                source=str(source),
                code="IDENTITY_CONFLICT",
                message=(
                    f"Effect ID {definition.effect_id!r} is already provided by {previous}; "
                    "the conflicting package was not loaded."
                ),
                effect_id=definition.effect_id,
            ))
            continue
        accepted.append(definition)
        occupied[definition.effect_id] = str(source)
    return LibraryMergeResult(EffectLibrary(accepted), tuple(diagnostics))


def unique_copy_definition(
    definition: EffectDefinitionV2,
    existing_ids: set[str] | frozenset[str] | tuple[str, ...],
) -> EffectDefinitionV2:
    """Return a semantically identical local copy with a collision-free identity."""

    occupied = {str(item) for item in existing_ids}
    counter = 2
    while True:
        candidate_id = f"{definition.effect_id}-copy-{counter}"
        if candidate_id not in occupied:
            break
        counter += 1
    return replace(
        definition,
        effect_id=candidate_id,
        name=f"{definition.name} Copy {counter}",
        built_in=False,
        package_origin="local-copy",
    )
