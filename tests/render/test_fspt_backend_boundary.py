"""The feasibility renderer must never silently use a CPU renderer."""
import importlib.util

import pytest


def test_cpu_device_is_refused_before_any_warp_import():
    assert importlib.util.find_spec("fluidSim_warp_blender51.fspt.backend"), "CUDA backend missing"
    from fluidSim_warp_blender51.fspt.backend import CudaBackend
    with pytest.raises(ValueError, match="CUDA"):
        CudaBackend("cpu")


def test_engine_is_part_of_the_addon_owned_registration_order():
    from fluidSim_warp_blender51.registration import STEPS
    matching = [step for step in STEPS if step.module == "blender.fspt_engine"]
    assert len(matching) == 1, "FSPT has no owned registration step"
    assert matching[0].register == "register"
    assert matching[0].unregister == "unregister"
