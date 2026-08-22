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


@lru_cache(maxsize=1)
def builtin_trust_digests() -> frozenset[str]:
    """Return exact content digests shipped by this installed FluidSim build."""

    from .builtins import builtin_effect_definitions

    return frozenset(content_trust_digest(item) for item in builtin_effect_definitions())


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
