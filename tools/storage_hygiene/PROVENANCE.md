# Vendored checker

`storage_hygiene.py`, `policy.json` and `tests/test_hygiene.py` are copied unchanged
from local shared commit `4b64d29dd0cc4707abdc28b452647f35963f6390`.
They require only Python's standard library and live in this repository; no dated
external task path, installation, network or optional import is needed.
`integration.py` binds them to Liquid's production entry point.

Update the checker and its regression suite together after independent review.
The shared policy's dated paths are retention/never-probe records, not executable
dependencies. Reconcile them with owners rather than stripping protections.
