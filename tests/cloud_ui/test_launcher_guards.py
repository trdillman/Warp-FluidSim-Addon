"""Fail-closed launcher guards; no Blender/UI process is launched by these tests."""
import os, subprocess, tempfile, unittest

import pytest

# The launcher is a Linux cloud script (flock); there is nothing to guard on Windows.
fcntl = pytest.importorskip("fcntl", reason="Linux-only cloud UI launcher")
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[2]/'tools/cloud_ui/run.sh'
class Guards(unittest.TestCase):
    def invoke(self, root, **extra):
        env=dict(os.environ,CLOUD_UI_WORK_ROOT=str(root),CLOUD_UI_RUN_ID='test',BLENDER_EXECUTABLE='/bin/true')
        env.update(extra)
        return subprocess.run(['bash',str(SCRIPT)],env=env,capture_output=True,text=True)
    def test_unsafe_run_id(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.invoke(Path(d),CLOUD_UI_RUN_ID='../escape'); self.assertNotEqual(r.returncode,0); self.assertIn('Unsafe run ID',r.stderr)
    def test_existing_run_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'test').mkdir(); r=self.invoke(root); self.assertNotEqual(r.returncode,0); self.assertIn('Run already exists',r.stderr)
    def test_wrong_binary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=self.invoke(Path(d)); self.assertNotEqual(r.returncode,0); self.assertIn('FAILED',r.stdout)
    def test_lock_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with (root/'.cloud-ui.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                r=self.invoke(root); self.assertNotEqual(r.returncode,0); self.assertIn('Another cloud UI lane',r.stderr)
if __name__=='__main__': unittest.main()
