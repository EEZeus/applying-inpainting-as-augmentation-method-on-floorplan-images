import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.augmentation.lora_audit import build_audit
from src.augmentation.lora_strategy_b import run_lora_strategy_b
from src.augmentation.backends import InpaintingBackend
from src.lora_inpainting.data import build_training_crops
from src.lora_inpainting.train import make_inpainting_unet_input
from src.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def ann(aid, category, image=1, bbox=(12, 12, 38, 38)):
    x,y,w,h=bbox
    return {"id":aid,"image_id":image,"category_id":category,"bbox":list(bbox),"area":w*h,"iscrowd":0,
            "segmentation":[[x,y,x+w,y,x+w,y+h,x,y+h]]}


def fixture(tmp_path):
    source=tmp_path/"source"; source.mkdir(); Image.new("RGB",(64,64),"white").save(source/"train.png")
    coco={"images":[{"id":1,"file_name":"train.png","width":64,"height":64},{"id":2,"file_name":"do-not-read.png","width":64,"height":64}],
          "annotations":[ann(1,12,bbox=(8,8,20,20)),ann(2,1,bbox=(35,35,20,20)),ann(3,12,image=2)],
          "categories":[{"id":i,"name":f"class-{i}"} for i in range(1,20)]}
    return coco,source


class Fake(InpaintingBackend):
    @property
    def metadata(self): return {"model_name":"fake","model_revision":"test","device":"cpu","gpu":"CPU","lora_sha256":"fake"}
    def generate(self,image,mask,prompt,seed,**kwargs):
        y,x=np.indices((image.height,image.width)); a=(180+(x+y)%50).astype(np.uint8)
        return Image.fromarray(np.repeat(a[...,None],3,axis=2))


def test_crop_manifest_separates_variants_from_geometries(tmp_path):
    coco,source=fixture(tmp_path); config=load_yaml(ROOT/"configs/lora_training.yaml")
    config["crop"]["output_size"]=64; config["mask"]["structure_protection"]=False
    config["rare_augmentation"]["primary_ids"]=[12]; config["rare_augmentation"]["optional_ids"]=[]
    config["rare_augmentation"]["variants_per_primary_instance"]=2
    summary=build_training_crops(config,coco,{1},source,tmp_path/"crops")
    manifest=json.loads((tmp_path/"crops/manifest.json").read_text())
    assert summary["real_crops"]==2 and summary["classical_variants"]==2
    assert {r["source_image_id"] for r in manifest["records"]}=={1}
    rare=[r for r in manifest["records"] if r["category_id"]==12]
    assert len({r["geometry_identity"] for r in rare})==1
    assert all(r["independent_geometry"] is False for r in rare)
    assert len(json.loads((tmp_path/"crops/validation_cases.json").read_text())["cases"])==1


def test_inpainting_input_has_nine_channels():
    import torch
    assert make_inpainting_unet_input(torch.zeros(2,4,8,8),torch.zeros(2,1,64,64),torch.zeros(2,4,8,8)).shape==(2,9,8,8)


def test_lora_candidate_is_integrated_by_automatic_qc(tmp_path):
    coco,source=fixture(tmp_path); config=load_yaml(ROOT/"configs/augmentation_lora_b.yaml")
    config["classes"]["target_ids"]=[12]; config["crop"]["output_size"]=64; config["geometry"]["tolerance"]=.9
    config["runtime"]["num_workers"]=0; config["generation"]["lora_path"]=str(tmp_path/"model")
    (tmp_path/"model").mkdir(); (tmp_path/"model/pytorch_lora_weights.safetensors").write_bytes(b"test")
    output,qa=tmp_path/"output",tmp_path/"qa"
    config["output"]["decision_audit_csv"]=str(qa/"decision_audit.csv")
    result=run_lora_strategy_b(config,coco,{}, {1},source,output,qa,backend=Fake())
    rows=build_audit(output,qa/"decision_audit.csv")
    assert rows[0]["status"]=="accepted"
    assert result["human_approval_used"] is False
    dataset=json.loads((output/"instances_train_augmented.json").read_text())
    synthetic=next(i for i in dataset["images"] if i.get("synthetic"))
    changed=[a for a in dataset["annotations"] if a["image_id"]==synthetic["id"]]
    assert next(a for a in changed if a["bbox"]==[35,35,20,20])["category_id"]==12
