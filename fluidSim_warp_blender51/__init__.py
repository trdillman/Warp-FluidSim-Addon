"""FluidSim Warp Blender extension registration.

Release metadata is owned by ``blender_manifest.toml`` and runtime identity by
``version.py``. Import-time registration neither probes nor installs runtime
dependencies: Blender installs the bundled, hash-locked NVIDIA Warp wheel with
the extension, and ``runtime/warp_runtime.py`` validates it lazily.
"""


def register():
    from . import registration

    registration.register_all(__name__)


def unregister():
    from . import registration

    registration.unregister_all(__name__)
