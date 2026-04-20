from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def save_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
