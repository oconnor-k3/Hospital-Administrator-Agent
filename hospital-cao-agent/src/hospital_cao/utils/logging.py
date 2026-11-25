# ---------------------------------------------------------------------------
# Generative AI disclosure: Claude (Anthropic) was used to help draft the code in this project, as well as debugging syntax issues if the code broke.# 
# ---------------------------------------------------------------------------

import csv
import json
from pathlib import Path


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_rows_csv(rows, path):
    """rows: list of flat dicts with the same keys."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

