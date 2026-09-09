#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.lora_inpainting.validation import validate_generator
from src.utils.config import load_yaml, resolve_path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default=ROOT/"configs/lora_training.yaml"); a=p.parse_args(); c=load_yaml(a.config)
    print(json.dumps(validate_generator(c, resolve_path(c["dataset"]["manifest"]), resolve_path(c["validation"]["output_dir"])), indent=2))
if __name__ == "__main__": main()
