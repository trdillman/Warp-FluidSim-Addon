from tools.gate.validate_source_acceptance_report import validate


def valid():
 return {"schema":"fluidsim-source-acceptance-v1","ok":True,"capability":"mesh_solid_volume","identity":{"blender_version":"5.1.0","blender_hash":"hash","addon_module":"fluidSim_warp_blender51","addon_version":[1,0,0],"addon_path":"/addon/__init__.py","device":"cuda:0","device_name":"GPU"},"fixture":{"name":"cube","blend_path":"/fixtures/cube.blend"},"measurements":{"active_voxels":64,"emitted_particles":0,"coverage_ratio":1.0,"attribute_checks":{"velocity":True,"temperature":True,"viscosity":True,"density":True,"lifespan":True},"details":{}},"invalid_input":{"case":"open","rejected":True,"message":"watertight required"},"timings_ms":{"compile":1.0,"step_median":1.0,"step_p95":1.0},"invariants":[{"id":"inside","passed":True,"expected":"inside","actual":"inside"}],"errors":[]}

def test_canonical_report_accepts_exact_types_and_all_passed(): assert validate(valid()) == []
def test_report_rejects_nonempty_errors_and_failed_invariant():
 report=valid()
 report["errors"]=["bad"]
 report["invariants"][0]["passed"]=False
 errors=validate(report)
 assert any("errors" in e for e in errors)
 assert any("invariant" in e for e in errors)

def test_report_rejects_false_or_empty_invalid_input_and_bad_details():
 report=valid()
 report["invalid_input"]={"case":"","rejected":False,"message":""}
 report["measurements"]["details"]=[]
 errors=validate(report)
 assert any("rejected" in e for e in errors)
 assert any("message" in e for e in errors)
 assert any("details" in e for e in errors)
