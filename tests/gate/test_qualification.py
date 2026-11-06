"""Qualification admission rejects changed inputs instead of blessing a HEAD label."""
import json
import subprocess

import pytest


def repo(tmp_path):
    for name in ('fluidSim_warp_blender51/sim/solver.py', 'tools/evaluator.py',
                 'demos/dam_break.blend', 'demos/crown_splash.blend'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '-qm', 'initial'], check=True)
    return tmp_path


def test_dirty_solver_cannot_be_admitted(tmp_path):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    (root / 'fluidSim_warp_blender51/sim/solver.py').write_text('dirty')
    with pytest.raises(RuntimeError, match='dirty product'):
        q.freeze_source(root)


@pytest.mark.parametrize('name', ['fluidSim_warp_blender51/sim/solver.py', 'tools/evaluator.py'])
def test_midrun_source_or_verifier_change_is_rejected(tmp_path, name):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    frozen = q.freeze_source(root)
    (root / name).write_text('changed')
    with pytest.raises(RuntimeError, match='changed during'):
        q.check_source(root, frozen)


def test_midrun_commit_is_rejected_even_when_product_is_identical(tmp_path):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    frozen = q.freeze_source(root)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.invalid', 'commit', '--allow-empty', '-qm', 'next'], check=True)
    with pytest.raises(RuntimeError, match='HEAD changed'):
        q.check_source(root, frozen)


def test_generated_exception_requires_matching_semantics_and_records_both_hashes(tmp_path):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    before = {'demos/dam_break.blend': 'a' * 64, 'demos/crown_splash.blend': 'b' * 64}
    frozen = q.freeze_source(root)
    (root / 'demos/dam_break.blend').write_text('different compression')
    with pytest.raises(RuntimeError, match='semantic'):
        q.check_source(root, frozen)
    changes = q.check_source(root, frozen, before, before)
    assert changes['demos/dam_break.blend']['before'] != changes['demos/dam_break.blend']['after']
    with pytest.raises(RuntimeError, match='semantic'):
        q.check_source(root, frozen, before, dict(before, **{'demos/dam_break.blend': 'c' * 64}))


def test_provenance_requires_reviewed_distribution_fields(tmp_path):
    from tools.gate import qualification as q
    file = tmp_path / 'provenance.json'
    file.write_text('{}')
    with pytest.raises(RuntimeError, match='distribution provenance'):
        q.read_provenance(file, 'a' * 64)


def test_legacy_and_dirty_measurement_receipts_are_never_reusable():
    from tools.gate import qualification as q
    for report in ({}, {'gitStatus': [' M solver.py']}):
        with pytest.raises(RuntimeError, match='provenance|dirty'):
            q.validate_binding(report, {})


@pytest.mark.parametrize('field', ['warp', 'gpu_uuid', 'driver', 'blender_sha256', 'platform'])
def test_changed_observed_environment_rejects_evidence(field):
    from tools.gate import qualification as q
    observed = {key: 'original' for key in ('warp', 'gpu_uuid', 'driver', 'blender_sha256', 'platform')}
    binding = {name: 'a' * 64 for name in ('source_fingerprint', 'verifier_fingerprint')}
    binding.update(environment_fingerprint=q.fingerprint(observed), run_id='measurement',
                   observed_environment=observed, gitStatus=[])
    changed = dict(observed, **{field: 'changed'})
    with pytest.raises(RuntimeError, match='environment'):
        q.validate_binding(binding, dict(binding, environment_fingerprint=q.fingerprint(changed)))


def test_wrong_warp_and_uncertified_blender_are_rejected():
    from tools.gate import qualification as q
    policy = {'blender_version': [5, 2, 0], 'warp_version': '1.19.0',
              'blender_sha256': 'a' * 64, 'python_sha256': 'b' * 64,
              'warp_wheel_sha256': 'c' * 64, 'python_prefix': '/blender/python'}
    observed = dict(policy, gpu_uuid='GPU-1', gpu_model='NVIDIA test', driver='600.1',
                    warp_distribution='1.19.0', warp_loaded_path='/site/warp/__init__.py',
                    blender_build='build', platform='Linux', architecture='x86_64',
                    python_version='3.13.0', python_executable='/blender/python/bin/python3.13')
    q.validate_environment(observed, policy)
    for field, value in (('warp_version', '1.17.0'), ('blender_version', [5, 1, 0]),
                         ('python_prefix', '/foreign/python')):
        with pytest.raises(RuntimeError, match=field):
            q.validate_environment(dict(observed, **{field: value}), policy)


def test_failed_storage_admission_invalidates_current_receipt_but_retains_evidence(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from tools import run_production_gate as gate
    from tools.storage_hygiene import integration
    manifest = tmp_path / 'artifacts/production_gate_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'run_id': 'earlier', 'passed': True}))
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    monkeypatch.setattr(gate, 'GATE_MANIFEST', manifest)
    @contextmanager
    def rejected(_root):
        raise RuntimeError('storage unavailable')
        yield
    monkeypatch.setattr(integration, 'production_storage', rejected)
    with pytest.raises(RuntimeError, match='storage unavailable'):
        gate.main(['--include-gui'])
    assert not manifest.exists()
    retained = list((tmp_path / 'artifacts/qualification_runs').glob('*/prior_manifest.json'))
    assert len(retained) == 1
    assert json.loads(retained[0].read_text())['run_id'] == 'earlier'


def test_uncertified_blender_is_rejected_before_wheel_or_gpu_probe(tmp_path, monkeypatch):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    (root / 'fluidSim_warp_blender51/version.py').write_text('CERTIFIED_BLENDER_VERSION=(5,2,0)\nCERTIFIED_WARP_VERSION="1.19.0"\nCERTIFIED_WARP_DISTRIBUTION="1.19.0+cu13"')
    executable = root / 'blender'
    executable.write_bytes(b'executable')
    interpreter = root / 'python/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b'python')
    approved = q.sha256(executable)
    provenance = root / 'provenance.json'
    provenance.write_text(json.dumps({'blender_executable_sha256': approved,
        'blender_archive_sha256': 'c' * 64, 'bundled_python_sha256': q.sha256(interpreter),
        'verifier_fingerprint': 'b' * 64,
        'blender_download_url': 'https://download.blender.org/release/Blender5.2/blender.tar.xz',
        'reviewed_by': 'Test reviewer'}))
    monkeypatch.setattr(q.subprocess, 'check_output', lambda *args, **kwargs: 'Blender 5.1.0\n')
    with pytest.raises(RuntimeError, match='certified Blender 5.2.0'):
        q.observe_runtime(root, executable, interpreter,
            {'BLENDER_SHA256': approved, 'BLENDER_PROVENANCE': str(provenance)}, root / 'evidence', 'b' * 64)


def test_midrun_edit_then_revert_still_invalidates_admission(tmp_path):
    from tools.gate import qualification as q
    root = repo(tmp_path)
    frozen = q.freeze_source(root)
    path = root / 'fluidSim_warp_blender51/sim/solver.py'
    old = path.read_bytes()
    path.write_bytes(b'transient edit')
    path.write_bytes(old)
    with pytest.raises(RuntimeError, match='changed during'):
        q.check_source(root, frozen)
