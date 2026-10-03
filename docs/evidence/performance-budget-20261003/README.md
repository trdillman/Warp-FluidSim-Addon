# D7 raw evidence

See ../../PERFORMANCE_BUDGET_GATE_20261003.md for the contract, result and recovery.
`runs.json.gz` preserves the exact raw JSON bytes (hash in budget.json).
Implementation hashes use UTF-8 text with LF line endings.

Recompute with the implementation in this change:

```python
import gzip, hashlib, json
from pathlib import Path
from tools.performance_budget import evaluate
root = Path("docs/evidence/performance-budget-20261003")
raw = gzip.decompress((root / "runs.json.gz").read_bytes())
report = json.loads((root / "budget.json").read_text())
assert hashlib.sha256(raw).hexdigest() == report["raw_sha256"]
result = evaluate(json.loads(raw))
assert result["demos"] == report["demos"]
assert result["verdict"] == report["verdict"] == "PASS"
```
