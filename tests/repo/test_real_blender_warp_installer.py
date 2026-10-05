"""Integration test proving the Preferences Install Warp button in real Blender 5.2.

Covers:
1. Offline refusal (--offline-mode and use_online_access=False).
2. Online install from PyPI via bpy.ops.fluidsim.install_warp():
   - Pinned Warp 1.17.0 lands strictly under the add-on's own extension path.
   - Blender's shared extensions/.local and user site-packages are never touched.
   - Runtime probe sees Warp 1.17.0.
   - Second press is idempotent and safe (no re-download, no rewrite).
3. Foreign Warp in shared extensions/.local:
   - Does not count as owned or already installed for FluidSim.
   - Refuses to overwrite or change foreign Warp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS_RELEASE = ROOT / "tools" / "release"
ADDON_ROOT = ROOT / "fluidSim_warp_blender51"


def _find_blender() -> Path | None:
    candidate = os.environ.get("BLENDER") or shutil.which("blender")
    if candidate and Path(candidate).is_file():
        return Path(candidate).resolve()
    for default in (
        Path("/usr/local/bin/blender"),
        Path("/opt/blender-5.2.0-linux-x64/blender"),
    ):
        if default.is_file():
            return default
    return None


@pytest.fixture(scope="module")
def built_package(tmp_path_factory) -> Path:
    blender = _find_blender()
    if blender is None:
        pytest.skip("Blender not found")

    sys.path.insert(0, str(TOOLS_RELEASE))
    try:
        import build_extension
    finally:
        sys.path.pop(0)

    dist_dir = tmp_path_factory.mktemp("real_blender_dist")
    packages = build_extension.build(
        ADDON_ROOT,
        blender,
        dist_dir,
        run_python_checks=False,
        run_lint=False,
    )
    linux_pkgs = [p for p in packages if "linux_x64" in p.name]
    assert len(linux_pkgs) == 1, f"Expected 1 linux package, got {packages}"
    return linux_pkgs[0]


def _setup_isolated_blender(base_dir: Path, package: Path, repo_id: str = "test_repo") -> dict[str, str]:
    blender = _find_blender()
    assert blender is not None

    repo_dir = base_dir / "repo"
    config_dir = base_dir / "config"
    scripts_dir = base_dir / "scripts"
    extensions_dir = base_dir / "extensions"
    datafiles_dir = base_dir / "datafiles"
    temp_dir = base_dir / "temp"

    for d in (repo_dir, config_dir, scripts_dir, extensions_dir, datafiles_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "BLENDER_USER_CONFIG": str(config_dir),
        "BLENDER_USER_SCRIPTS": str(scripts_dir),
        "BLENDER_USER_EXTENSIONS": str(extensions_dir),
        "BLENDER_USER_DATAFILES": str(datafiles_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
    })

    subprocess.run(
        [
            str(blender),
            "--command",
            "extension",
            "repo-add",
            repo_id,
            "--name",
            "Test Repo",
            "--directory",
            str(repo_dir),
            "--clear-all",
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        [
            str(blender),
            "--command",
            "extension",
            "install-file",
            "-r",
            repo_id,
            "-e",
            str(package),
        ],
        env=env,
        check=True,
        capture_output=True,
    )

    return env


def test_offline_refusal_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify that bpy.ops.fluidsim.install_warp() refuses before download when offline."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    script = """
import bpy
from pathlib import Path

assert bpy.app.online_access is False
module = 'bl_ext.test_repo.fluidsim_warp'
owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))

try:
    res = bpy.ops.fluidsim.install_warp()
    raise AssertionError(f'Expected operator error/cancellation, got {res}')
except RuntimeError as exc:
    assert 'Online access required' in str(exc), f'Unexpected message: {exc}'

assert not (owned_dir / 'warp').exists(), 'Warp directory was created during offline refusal!'
print('OFFLINE-REFUSE-VERIFIED')
"""
    result = subprocess.run(
        [
            str(blender),
            "--background",
            "--offline-mode",
            "--python-expr",
            script,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "OFFLINE-REFUSE-VERIFIED" in result.stdout


def test_online_install_and_idempotency_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify online install, destination isolation, probe detection, and second-press idempotency."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    script = """
import bpy
import site
import importlib
from pathlib import Path

bpy.context.preferences.system.use_online_access = True
assert bpy.app.online_access is True

module = 'bl_ext.test_repo.fluidsim_warp'
owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
shared_extensions = Path(bpy.utils.user_resource('EXTENSIONS'))
shared_local = shared_extensions / '.local'
user_site = Path(site.getusersitepackages())

assert not (owned_dir / 'warp').exists()

# 1. First press: online install from PyPI
res = bpy.ops.fluidsim.install_warp()
assert res == {'FINISHED'}, f'Expected FINISHED, got {res}'

warp_init = owned_dir / 'warp' / '__init__.py'
assert warp_init.is_file(), f'Expected Warp installed at {warp_init}'

# Shared extensions/.local and user site-packages must NEVER be touched
assert not (shared_local / 'warp').exists(), f'Shared {shared_local} was touched!'
assert not (user_site / 'warp').exists(), f'User site {user_site} was touched!'

# 2. Runtime probe detection
res_probe = bpy.ops.fluidsim.refresh_runtime_probe()
assert res_probe == {'FINISHED'}

runtime = importlib.import_module(f'{module}.runtime.warp_runtime')
probe = runtime.probe_runtime()
assert probe.warp_version == '1.17.0', f'Expected 1.17.0, got {probe.warp_version}'
assert Path(probe.module_file).resolve() == warp_init.resolve()

# 3. Second press: idempotent, no rewrite or foreign changes
stat_before = warp_init.stat()
res2 = bpy.ops.fluidsim.install_warp()
assert res2 == {'FINISHED'}, f'Expected FINISHED, got {res2}'
stat_after = warp_init.stat()

assert stat_before.st_mtime == stat_after.st_mtime, 'Warp was rewritten on second press!'
assert not (shared_local / 'warp').exists()
assert not (user_site / 'warp').exists()

print('ONLINE-INSTALL-AND-IDEMPOTENCY-VERIFIED')
"""
    result = subprocess.run(
        [
            str(blender),
            "--background",
            "--python-expr",
            script,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "ONLINE-INSTALL-AND-IDEMPOTENCY-VERIFIED" in result.stdout


def test_shared_extensions_local_foreign_warp_refusal_in_real_blender(tmp_path: Path, built_package: Path):
    """Verify that Warp in shared extensions/.local is not treated as owned and is not overwritten."""
    blender = _find_blender()
    env = _setup_isolated_blender(tmp_path, built_package)

    # Seed Warp into shared extensions/.local
    shared_local = Path(env["BLENDER_USER_EXTENSIONS"]) / ".local"
    warp_dir = shared_local / "warp"
    warp_dir.mkdir(parents=True, exist_ok=True)
    (warp_dir / "__init__.py").write_text("__version__ = '1.17.0'\n", encoding="utf-8")
    dist_info = shared_local / "warp_lang-1.17.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text("Name: warp-lang\nVersion: 1.17.0\n", encoding="utf-8")

    script = """
import bpy
import sys
from pathlib import Path

bpy.context.preferences.system.use_online_access = True
module = 'bl_ext.test_repo.fluidsim_warp'
installer = __import__(f'{module}.warp_installer', fromlist=['warp_installer'])

shared_local = Path(bpy.utils.user_resource('EXTENSIONS')) / '.local'
assert (shared_local / 'warp' / '__init__.py').is_file()

owned_dir = Path(bpy.utils.extension_path_user(module, path='.local', create=False))
assert not (owned_dir / 'warp').exists()

# Shared extensions/.local must NOT be in owned candidates or satisfy installed check
assert shared_local not in installer._owned_candidates()
assert installer.owned_install_status() is None
assert installer.is_certified_warp_installed() is False

# If foreign Warp was imported from shared_local, install_warp must refuse
if str(shared_local) not in sys.path:
    sys.path.insert(0, str(shared_local))
import warp as wp
assert wp.__file__ == str(shared_local / 'warp' / '__init__.py')

try:
    bpy.ops.fluidsim.install_warp()
    raise AssertionError('Expected refusal when foreign Warp is loaded')
except RuntimeError as exc:
    assert 'which FluidSim does not own' in str(exc)
    assert 'extensions/.local' in str(exc)

print('SHARED-EXTENSIONS-LOCAL-REFUSAL-VERIFIED')
"""
    result = subprocess.run(
        [
            str(blender),
            "--background",
            "--python-expr",
            script,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Blender failed: {result.stderr}\n{result.stdout}"
    assert "SHARED-EXTENSIONS-LOCAL-REFUSAL-VERIFIED" in result.stdout
