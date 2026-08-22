"""Real Warp compilation for Custom Effects v2."""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .enums import TrustState
from .schema import EffectInstanceV2, EffectIssue
from .fingerprint import effective_stack_instances
from .source import GeneratedKernelSource, generate_kernel_source
from .trust import TrustStore, content_trust_digest
from .validation import validate_effect


@dataclass(frozen=True)
class CompileReport:
    compiled: bool
    device: str
    warp_version: str
    module_file: str
    source_digest: str
    duration_seconds: float


@dataclass(frozen=True)
class CompiledKernel:
    generated: GeneratedKernelSource
    kernel_definition: object
    kernel: object
    module: object
    compile_report: CompileReport


@dataclass(frozen=True)
class CompiledEffect:
    instance: EffectInstanceV2
    success: bool
    kernels: tuple[CompiledKernel, ...]
    issues: tuple[EffectIssue, ...]


@dataclass(frozen=True)
class CandidateBuildResult:
    success: bool
    compiled_effects: tuple[CompiledEffect, ...]
    sandbox_reports: tuple[object, ...]
    issues: tuple[EffectIssue, ...]
    trust_states: dict[str, TrustState]


def _import_generated(generated: GeneratedKernelSource, cache_dir: str | Path | None):
    path = generated.write_cache_file(cache_dir)
    if generated.module_name in sys.modules:
        module = sys.modules[generated.module_name]
        if Path(getattr(module, "__file__", "")) == path:
            return module, path
        del sys.modules[generated.module_name]
    spec = importlib.util.spec_from_file_location(generated.module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[generated.module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(generated.module_name, None)
        raise
    return module, path


def _compile_issue(instance: EffectInstanceV2, generated: GeneratedKernelSource, exc: Exception) -> EffectIssue:
    message = f"{type(exc).__name__}: {exc}"
    line = 0
    import re
    candidates = [
        re.search(r"line\s+(\d+)", message, re.IGNORECASE),
        re.search(r":(\d+)(?::\d+)?", message),
    ]
    for match in candidates:
        if match:
            line = generated.original_line_for_generated(int(match.group(1)))
            if line:
                break
    return EffectIssue(
        "WARP_COMPILE_FAILED",
        message,
        effect_id=instance.definition.effect_id,
        instance_id=instance.instance_id,
        kernel_id=generated.kernel_id,
        line=line,
        hint="Open the source in Blender's Text Editor and inspect the mapped line.",
    )


def compile_effect(
    instance: EffectInstanceV2,
    *,
    device: str = "cpu",
    cache_dir: str | Path | None = None,
) -> CompiledEffect:
    validation = validate_effect(instance)
    if not validation.valid:
        return CompiledEffect(instance, False, (), validation.issues)

    import warp as wp

    kernels: list[CompiledKernel] = []
    issues: list[EffectIssue] = []
    for index, definition in enumerate(instance.definition.kernels):
        generated = generate_kernel_source(instance, index)
        started = time.perf_counter()
        try:
            module, path = _import_generated(generated, cache_dir)
            wp.load_module(module=module, device=device)
            # Warp suppresses repeated failed builds by returning ``None`` on
            # subsequent loads.  Inspect the concrete module execution so a
            # previous JIT failure can never become a false positive.
            warp_module = wp.get_module(module.__name__)
            module_exec = warp_module.load(wp.get_device(device))
            if module_exec is None:
                raise RuntimeError(
                    "Warp module did not produce an executable; a prior JIT build failed"
                )
            wp.synchronize_device(device)
            kernel = getattr(module, generated.kernel_name)
            report = CompileReport(
                compiled=True,
                device=str(device),
                warp_version=str(getattr(wp, "__version__", "unknown")),
                module_file=str(path),
                source_digest=generated.source_digest,
                duration_seconds=max(0.0, time.perf_counter() - started),
            )
            kernels.append(CompiledKernel(generated, definition, kernel, module, report))
        except Exception as exc:  # noqa: BLE001 - converted to structured diagnostics
            issues.append(_compile_issue(instance, generated, exc))
            break
    if issues:
        return CompiledEffect(instance, False, (), tuple(issues))
    return CompiledEffect(instance, True, tuple(kernels), ())


def build_candidate_effects(
    instances,
    *,
    device: str,
    trust_store: TrustStore,
) -> CandidateBuildResult:
    from .sandbox import sandbox_test

    trust_states: dict[str, TrustState] = {}
    issues: list[EffectIssue] = []
    try:
        active = list(effective_stack_instances(tuple(instances)))
    except ValueError as exc:
        issue = EffectIssue("INVALID_EFFECT_STACK", str(exc))
        return CandidateBuildResult(False, (), (), (issue,), {})
    for instance in active:
        digest = content_trust_digest(instance.definition)
        state = trust_store.status(digest, built_in=instance.definition.built_in)
        trust_states[instance.instance_id] = state
        if state not in {TrustState.BUILT_IN_TRUSTED, TrustState.LOCAL_TRUSTED}:
            issues.append(EffectIssue(
                "EFFECT_UNTRUSTED" if state is not TrustState.BLOCKED else "EFFECT_BLOCKED",
                f"Effect {instance.definition.name!r} is {state.value.lower().replace('_', ' ')}.",
                effect_id=instance.definition.effect_id,
                instance_id=instance.instance_id,
                hint="Review and trust the exact content digest before applying it.",
            ))
    if issues:
        return CandidateBuildResult(False, (), (), tuple(issues), trust_states)

    compiled: list[CompiledEffect] = []
    reports: list[object] = []
    for instance in active:
        result = compile_effect(instance, device=device)
        if not result.success:
            issues.extend(result.issues)
            return CandidateBuildResult(False, (), (), tuple(issues), trust_states)
        report = sandbox_test(result, device=device)
        if not report.passed:
            issues.extend(report.issues)
            return CandidateBuildResult(False, (), (), tuple(issues), trust_states)
        compiled.append(result)
        reports.append(report)
    return CandidateBuildResult(True, tuple(compiled), tuple(reports), (), trust_states)
