"""Qualification admission rejects changed inputs instead of blessing a HEAD label."""
import json
import subprocess
from pathlib import Path

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
