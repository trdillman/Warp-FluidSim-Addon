"""Mandatory policy failure and warning compatibility; fixtures only."""
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

class AdmissionTests(unittest.TestCase):
    def test_missing_required_policy_blocks_body_with_admission_code(self):
        called = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(helper, 'HERE', Path(directory)), \
             patch.object(helper.subprocess, 'run'), \
             patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                with helper.production_storage(Path(directory)):
                    called.append(True)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(called, [])
        self.assertIn('STORAGE-HYGIENE-DEPENDENCY-FAIL', output.getvalue())

    def test_default_policy_snapshot_is_not_read_twice(self):
        target = helper.HERE / 'policy.json'
        original = Path.read_text
        count = []
        def replaced(path, *args, **kwargs):
            if path == target:
                count.append(True)
                if len(count) > 1:
                    return json.dumps(dict(version=1, drive=target.anchor, roots=[], never_probe=[],
                        free_warn_bytes=0, max_entries=100, max_seconds=1))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', new=replaced):
            config = helper.policy(Path.cwd(), default=True)
        self.assertEqual(len(config['roots']), 4)
        self.assertEqual(len(count), 1)

    def test_empty_but_valid_required_policy_does_not_establish_telemetry(self):
        called = []
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'policy.json').write_text(json.dumps(dict(version=1, drive=base.anchor,
                roots=[], never_probe=[], free_warn_bytes=0, max_entries=100, max_seconds=1)))
            with patch.object(helper, 'HERE', base), patch.object(helper.subprocess, 'run'), \
                 patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    with helper.production_storage(base):
                        called.append(True)
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(called, [])

    def config(self, directory):
        root = Path(directory)
        return dict(version=1, drive=root.anchor, roots=[dict(path=str(root), budget_bytes=100, stale_days=7,
            owner='test', project='fixture', state='retained', disposable=False, retention_reason='fixture', protected=[])],
            never_probe=[], free_warn_bytes=10**18, max_entries=0, max_seconds=1)

    def test_unavailable_measurement_blocks_body(self):
        called = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(helper, 'policy', return_value=self.config(directory)), \
             patch.object(helper.subprocess, 'run'), \
             patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
             patch.object(helper.checker, 'check', side_effect=RuntimeError('unexpected measurement failure')), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as caught:
                with helper.production_storage(Path(directory)):
                    called.append(True)
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(called, [])
        self.assertIn('STORAGE-HYGIENE-DEPENDENCY-FAIL', output.getvalue())

    def test_expected_incomplete_and_low_free_remain_warnings(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(helper, 'policy', return_value=self.config(directory)), \
             patch.object(helper.subprocess, 'run'), \
             patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with helper.production_storage(Path(directory)):
                pass
        self.assertIn('coverage_incomplete', output.getvalue())
        self.assertIn('low_free_space', output.getvalue())
        self.assertIn('phase=end-task', output.getvalue())

    def test_end_failure_preserves_production_exception(self):
        emit = helper.checker.emit
        error = RuntimeError('production fixture')
        def failure_at_end(config, phase, project):
            return None if phase == 'end-task' else emit(config, phase, project)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(helper, 'policy', return_value=self.config(directory)), \
             patch.object(helper.subprocess, 'run'), \
             patch.dict('os.environ', {'FLUIDSIM_STORAGE_POLICY': ''}), \
             patch.object(helper.checker, 'emit', side_effect=failure_at_end), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                with helper.production_storage(Path(directory)):
                    raise error
        self.assertIs(caught.exception, error)
