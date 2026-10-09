"""FluidSim Warp Blender extension registration.

Release metadata is owned by ``blender_manifest.toml`` and runtime identity by
``version.py``. Import-time registration neither probes nor installs runtime
dependencies: users install NVIDIA Warp via a button in Add-on Preferences,
and ``runtime/warp_runtime.py`` validates it lazily. Registration schedules
a deferred prewarm; first installation reschedules it once CUDA is ready.
"""


def register():
    from . import registration

    registration.register_all(__name__)


def unregister():
    from . import registration

    registration.unregister_all(__name__)
