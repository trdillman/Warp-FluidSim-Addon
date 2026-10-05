"""Synthetic integration tests: no production gate or real container scans."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.storage_hygiene import integration as helper

class SuccessorTests(unittest.TestCase):
    def test_default_four_container_config_survives_override(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'policy.json'
            policy.write_text(json.dumps(dict(version=1, drive=policy.anchor, free_warn_bytes=0,
                max_entries=0, max_seconds=0, roots=[], never_probe=[])))
            with patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': str(policy)}):
                default = helper.policy(Path(directory), default=True)
                local = helper.policy(Path(directory))
                seen = []
                with patch.object(helper, 'emit_configs', side_effect=lambda configs, phase: seen.append((phase, configs)) or True), \
                     patch.object(helper.subprocess, 'run'):
                    with helper.production_storage(Path(directory)):
                        pass
            self.assertEqual(len(default['roots']), 4)
            self.assertEqual(local['roots'], [])
            self.assertEqual([phase for phase, _ in seen], ['pre-production', 'end-task'])
            self.assertEqual([len(configs[0][1]['roots']) for _, configs in seen], [4, 4])
            self.assertTrue(set(default['never_probe']).issubset(local['never_probe']))

    def test_failure_keeps_exception_and_end_report(self):
        cfg = dict(version=1, drive=str(Path.cwd().anchor), free_warn_bytes=0,
            max_entries=100, max_seconds=1, roots=[], never_probe=[])
        with patch.object(helper, 'policy', return_value=cfg), \
             patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
             patch.object(helper.subprocess, 'run'), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, 'production fixture'):
                with helper.production_storage(Path.cwd()):
                    raise RuntimeError('production fixture')
        self.assertIn('phase=end-task', output.getvalue())

    def test_override_holds_apply_before_global_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            hold = str(base / 'hold')
            denied = str(base / 'denied')
            default = dict(version=1, drive=base.anchor, free_warn_bytes=0,
                           max_entries=100, max_seconds=1, roots=[], never_probe=[])
            local = dict(default, roots=[dict(path=str(base), budget_bytes=100, stale_days=7,
                project='fixture', owner='test', state='retained', disposable=False,
                retention_reason='fixture', protected=[hold])], never_probe=[denied])
            def policy(root, *, default=False):
                return globals_config if default else local
            globals_config = default
            seen = []
            with patch.object(helper, 'policy', side_effect=policy), \
                 patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': 'synthetic-policy'}), \
                 patch.object(helper, 'emit_configs', side_effect=lambda configs, phase: seen.append(configs) or True), \
                 patch.object(helper.subprocess, 'run'):
                with helper.production_storage(base):
                    pass
            for _, config in seen[0]:
                self.assertTrue({hold, denied}.issubset(config['never_probe']))

    def test_denials_carried_to_both_configs(self):
        configs = [('liquid', {'never_probe': []}), ('liquid:owned', {'never_probe': []})]
        path = str(Path.cwd() / 'synthetic-denied')
        result = {'roots': [{'inaccessible_paths': [path]}], 'warnings': []}
        with patch.object(helper.checker, 'emit', return_value=result):
            helper.emit_configs(configs, 'pre-production')
        self.assertTrue(all(config['never_probe'] == [path] for _, config in configs))
