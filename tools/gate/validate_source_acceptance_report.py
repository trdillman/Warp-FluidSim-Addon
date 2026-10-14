#!/usr/bin/env python3
"""Validate canonical Phase 2 source acceptance JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SCHEMA = "fluidsim-source-acceptance-v1"
TOP = {"schema","ok","capability","identity","fixture","measurements",
       "invalid_input","timings_ms","invariants","errors"}
IDENTITY = {"blender_version","blender_hash","addon_module","addon_version",
            "addon_path","device","device_name"}
MEASUREMENTS = {"active_voxels","emitted_particles","coverage_ratio",
                "attribute_checks","details"}
ATTRS = {"velocity","temperature","viscosity","density","lifespan"}

def validate(report):
    errors=[]
    if set(report) != TOP:
        errors.append(f"top-level keys: {sorted(set(report)^TOP)}")
    if report.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if report.get("ok") is not True:
        errors.append("ok must be true")
    if not isinstance(report.get("capability"),str):
        errors.append("capability must be string")
    identity=report.get("identity",{})
    if set(identity)!=IDENTITY:
        errors.append("identity keys mismatch")
    if not isinstance(identity.get("addon_version"),list) or not all(isinstance(v,int) for v in identity.get("addon_version",[])):
        errors.append("addon_version must be integer array")
    for key in IDENTITY-{"addon_version"}:
        if not isinstance(identity.get(key),str):
            errors.append(f"identity.{key} must be string")
    measurements=report.get("measurements",{})
    if not MEASUREMENTS.issubset(measurements):
        errors.append("measurement keys missing")
    for key in ("active_voxels","emitted_particles"):
        if measurements.get(key) is not None and not isinstance(measurements.get(key),int):
            errors.append(f"{key} must be int/null")
    if measurements.get("coverage_ratio") is not None and not isinstance(measurements.get("coverage_ratio"),(int,float)):
        errors.append("coverage_ratio must be number/null")
    attrs=measurements.get("attribute_checks")
    if attrs is not None and (not isinstance(attrs,dict) or set(attrs)!=ATTRS or not all(isinstance(v,bool) for v in attrs.values())):
        errors.append("attribute_checks invalid")
    if not isinstance(measurements.get("details"),dict):
        errors.append("measurements.details must be object")
    invalid=report.get("invalid_input",{})
    if set(invalid)!={"case","rejected","message"}:
        errors.append("invalid_input invalid")
    else:
        if invalid.get("rejected") is not True:
            errors.append("invalid_input.rejected must be true")
        if not isinstance(invalid.get("case"),str) or not invalid.get("case").strip():
            errors.append("invalid_input.case must be non-empty")
        if not isinstance(invalid.get("message"),str) or not invalid.get("message").strip():
            errors.append("invalid_input.message must be actionable")
    timings=report.get("timings_ms",{})
    if set(timings)!={"compile","step_median","step_p95"} or not all(isinstance(v,(int,float)) for v in timings.values()):
        errors.append("timings invalid")
    invariants=report.get("invariants")
    if not isinstance(invariants,list) or not invariants:
        errors.append("invariants must be non-empty array")
    elif any(set(v)!={"id","passed","expected","actual"} or v.get("passed") is not True for v in invariants):
        errors.append("invariant failed/invalid")
    if report.get("errors") != []:
        errors.append("errors must be empty")
    return errors

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("report")
    args=parser.parse_args(argv)
    report=json.loads(Path(args.report).read_text(encoding="utf8"))
    errors=validate(report)
    if errors:
        raise SystemExit("SOURCE-ACCEPTANCE-REPORT-FAIL: "+"; ".join(errors))
    print("SOURCE-ACCEPTANCE-REPORT-PASS")
if __name__=="__main__":
    main()
