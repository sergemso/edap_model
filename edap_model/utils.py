"""Constants and utilities for EDAP model."""

import json
import os
import numpy as np

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_DIR = "output"


def load_json(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    if os.path.getsize(filename) == 0:
        raise ValueError(f"File is empty: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            raise ValueError(f"File contains only whitespace: {filename}")
        return json.loads(content)
