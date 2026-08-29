"""Sibling Warp release validators expose one consistent Python API."""

from __future__ import annotations

import inspect

from fluidsim_testpkg.release import (
    validate_phase3_warp,
    validate_phase4_warp,
    validate_phase5_warp,
)


def test_warp_validators_default_to_the_validation_package():
    for validator in (
        validate_phase3_warp,
        validate_phase4_warp,
        validate_phase5_warp,
    ):
        parameter = inspect.signature(validator.validate).parameters["module_name"]
        assert parameter.default == validator.VALIDATION_PACKAGE
