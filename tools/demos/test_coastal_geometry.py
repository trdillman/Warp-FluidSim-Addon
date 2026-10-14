from pathlib import Path
import ast, numpy as np, math, json,sys
root=Path(r'F:\LiquidWarp-CoastalCascade-20261011')
src=(root/'source/tools/demos/build_coastal_cascade.py').read_text(encoding='utf8')
tree=ast.parse(src);ns={'np':np,'math':math}
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ('center','width','smooth','bed_level','bed_height'):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<geometry>','exec'),ns)
checks={'central_stack_is_in_water_not_buried':float(ns['bed_height'](1.3,4.5))<1.0,'open_right_bay':float(ns['bed_height'](8.,4.5))<1.0,'no_invalid_quit_keyword':'quit_blender(confirm=False)' not in src,'cold_reopen_test_provided':(root/'source/tools/demos/verify_coastal_cascade.py').exists()}
report={'checks':checks,'bed_heights':{str(p):float(ns['bed_height'](*p)) for p in [(1.3,4.5),(8,4.5),(-10,20),(-3,-5)]}}
print(json.dumps(report,indent=2));(root/'evidence/geometry_contract_latest.json').write_text(json.dumps(report,indent=2));sys.exit(0 if all(checks.values()) else 1)