#!/usr/bin/env python3
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.experiments.report import generate_final_report

if __name__ == "__main__":
    print(json.dumps(generate_final_report(), indent=2))
