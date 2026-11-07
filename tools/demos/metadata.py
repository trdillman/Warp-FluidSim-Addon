"""Write generated demo metadata without rewriting identical source inputs."""

import json
from pathlib import Path


def write_metadata(path, metadata):
    path = Path(path)
    content = json.dumps(metadata, indent=2) + "\n"
    try:
        if path.read_text(encoding="utf-8") == content:
            return
    except FileNotFoundError:
        pass
    path.write_text(content, encoding="utf-8", newline="\n")
