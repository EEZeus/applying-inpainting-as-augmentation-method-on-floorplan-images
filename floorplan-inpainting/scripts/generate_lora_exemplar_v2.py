#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.augmentation.inputs import load_inputs
from src.augmentation.strategy_c import run_strategy_c
from src.utils.config import load_yaml,resolve_path
from src.utils.reproducibility import seed_everything
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=ROOT/"configs/augmentation_lora_exemplar_v2.yaml");p.add_argument("--retry-errors",action="store_true");a=p.parse_args()
    c=load_yaml(a.config);c["generation"]["cache_dir"]=str(resolve_path(c["generation"]["cache_dir"]));c["generation"]["lora_path"]=str(resolve_path(c["generation"]["lora_path"]))
    coco,rare,train,image_dir=load_inputs(c);seed_everything(c["seed"])
    result=run_strategy_c(c,coco,rare,train,image_dir,resolve_path(c["output"]["dataset_dir"]),resolve_path(c["output"]["qa_dir"]),"diffusion_inpaint",retry_errors=a.retry_errors)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
