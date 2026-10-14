"""Local content-digest trust storage for executable custom effects."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .enums import TrustState
from .fingerprint import sha256_hex
from .schema import EffectDefinitionV2

_TRUST_SCHEMA_VERSION = 1


def content_trust_digest(definition: EffectDefinitionV2) -> str:
    payload = definition.to_dict()
    payload.pop("built_in", None)
    payload.pop("package_origin", None)
    payload.pop("preview_asset", None)
    return sha256_hex({"trust_content_version": 1, "definition": payload})


# Exact content digests of built-in definitions earlier FluidSim releases
# shipped. A .blend embeds the definition it was authored with, so these stay
# trusted as built-ins after the shipped content changes; only the built_in
# flag together with an exact digest grants it. Never remove an entry.
RETIRED_BUILTIN_DIGESTS = frozenset({
    # Built-ins 1.0.0 (through FluidSim 3.1.0): particle-velocity kernels at
    # AFTER_PARTICLE_TRANSFER; voxel_cooling before its documentation fix (B68).
    "f8fa4367589ce6bd607b52e9d1972053fe49c65783b07f892b78ba96ab9a4a9e",  # vortex_force
    "84f267386b36a42a8242e3371eec93de6caffcd66d7a21be098db08dc77c6c8b",  # curl_turbulence
    "37ea7cc389a99398315cf114bc6b47bf4f575c60a1d943cbd27ce8f4a270e557",  # radial_pulse
    "c24ba5a2c167352057f08806663ff2727330fb4007a0404b35355e9491849f1f",  # planar_split
    "08294a474fe956f00fdf5f64d22c09355c4dbf49f2cdb4057c2d03e5ba9b4bdb",  # velocity_damping
    "4bd7c1c4cc97a39994da8382e2e840f93ce2258fb527e0bd41c7f28e28183f20",  # temperature_lift
    "a9f2ee0c14e4dfec878b4267801822827d9f6ad706f794e10c9495264a1d3b4f",  # particle_speed_limit
    "7cc5a78b20fe30ea4638f1e45eaea62c5c28e652f8c3984f16fb515e4472309e",  # voxel_cooling
})


@lru_cache(maxsize=1)
def builtin_trust_digests() -> frozenset[str]:
    """Return exact content digests shipped by this or an earlier FluidSim build."""

    from .builtins import builtin_effect_definitions

    current = frozenset(content_trust_digest(item) for item in builtin_effect_definitions())
    return current | RETIRED_BUILTIN_DIGESTS


class TrustStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid effect trust store: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != _TRUST_SCHEMA_VERSION:
            raise ValueError("unsupported effect trust-store schema")
        records = payload.get("records")
        if not isinstance(records, dict):
            raise ValueError("effect trust-store records must be an object")
        normalized: dict[str, dict[str, str]] = {}
        for digest, record in records.items():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("invalid trust-store digest")
            if not isinstance(record, dict):
                raise ValueError("invalid trust-store record")
            state = str(record.get("state", ""))
            if state not in {TrustState.LOCAL_TRUSTED.value, TrustState.BLOCKED.value}:
                raise ValueError(f"invalid trust-store state {state!r}")
            normalized[digest] = {
                "state": state,
                "label": str(record.get("label", "")),
                "reason": str(record.get("reason", "")),
            }
        return normalized

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        payload = {
            "schema_version": _TRUST_SCHEMA_VERSION,
            "records": {key: self._records[key] for key in sorted(self._records)},
        }
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.path)
        try:
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def status(self, digest: str, *, built_in: bool = False) -> TrustState:
        if built_in and digest in builtin_trust_digests():
            return TrustState.BUILT_IN_TRUSTED
        record = self._records.get(digest)
        if record is None:
            return TrustState.UNTRUSTED_IMPORTED
        return TrustState(record["state"])

    def trust(self, digest: str, label: str) -> None:
        self._records[digest] = {
            "state": TrustState.LOCAL_TRUSTED.value,
            "label": str(label),
            "reason": "",
        }
        self._write()

    def block(self, digest: str, reason: str) -> None:
        self._records[digest] = {
            "state": TrustState.BLOCKED.value,
            "label": self._records.get(digest, {}).get("label", ""),
            "reason": str(reason),
        }
        self._write()

    def revoke(self, digest: str) -> None:
        if digest in self._records:
            del self._records[digest]
            self._write()
