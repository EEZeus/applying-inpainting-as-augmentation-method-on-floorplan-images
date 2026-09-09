import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.augmentation.lora_multiroom import build_multiroom_dataset
from src.augmentation.masks import build_safe_mask
from src.utils.config import load_yaml
ROOT=Path(__file__).resolve().parents[1]
def ann(aid,category,bbox):
 x,y,w,h=bbox;return {"id":aid,"image_id":1,"category_id":category,"bbox":list(bbox),"area":w*h,"iscrowd":0,"segmentation":[[x,y,x+w,y,x+w,y+h,x,y+h]]}
def test_long_line_mode_edits_text_but_protects_structural_line():
 image=np.full((100,120,3),255,np.uint8);cv2.line(image,(22,35),(77,35),(0,0,0),1);cv2.putText(image,"BURO",(35,60),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
 target=ann(1,1,(20,20,60,60));result=build_safe_mask(image,target,[],{"structure_protection":True,"structure_protection_mode":"long_lines","dark_threshold":100,"structure_dilation_px":0})
 assert not result.safe_mask[35,30]  # long structural stroke protected
 assert result.safe_mask[57:62,35:62].any()  # compact semantic text is editable
def test_multiroom_compositor_enforces_common_to_genuine_rare(tmp_path):
 source=tmp_path/"source";source.mkdir();original=np.full((50,60,3),255,np.uint8);Image.fromarray(original).save(source/"x.png")
 candidate=tmp_path/"candidates"; (candidate/"full_images").mkdir(parents=True);changed=original.copy();changed[15:25,15:25]=100;Image.fromarray(changed).save(candidate/"full_images/c.png")
 record={"sample_id":"c","status":"accepted","source_image_id":1,"source_category_id":1,"category_id":11,"donor_category_id":11,"donor_annotation_id":2,"target_annotation_id":1,"output_file":"full_images/c.png","semantic_correctness":"real_train_exemplar_preserved"}
 (candidate/"metadata.jsonl").write_text(json.dumps(record)+"\n")
 coco={"images":[{"id":1,"file_name":"x.png","width":60,"height":50}],"annotations":[ann(1,1,(10,10,20,20)),ann(2,11,(35,10,15,15))],"categories":[{"id":1,"name":"common"},{"id":11,"name":"rare"}]}
 config=load_yaml(ROOT/"configs/augmentation_lora_multi.yaml");config["classes"]={"source_ids":[1],"target_ids":[11]};config["multi_room"]={"replacement_percentage":50,"random_jitter_percentage_points":0}
 manifest=build_multiroom_dataset(config,coco,{1},source,candidate,tmp_path/"out")
 assert manifest["all_sources_are_common"] and manifest["all_targets_and_donors_are_rare"]
 output=json.loads((tmp_path/"out/instances_train_augmented.json").read_text());synthetic=next(i for i in output["images"] if i.get("synthetic"));converted=[a for a in output["annotations"] if a["image_id"]==synthetic["id"] and a["bbox"]==[10,10,20,20]][0]
 assert converted["category_id"]==11
def test_multiroom_compositor_supports_strategy_c_copy_paste(tmp_path):
 source=tmp_path/"source";source.mkdir();original=np.full((40,60,3),255,np.uint8);Image.fromarray(original).save(source/"x.png")
 candidate=tmp_path/"candidates";(candidate/"full_images").mkdir(parents=True);changed=original.copy();changed[10:20,10:20]=80;Image.fromarray(changed).save(candidate/"full_images/c.png")
 record={"sample_id":"c","status":"accepted","source_image_id":1,"source_category_id":1,"category_id":11,"donor_category_id":11,"donor_annotation_id":2,"target_annotation_id":1,"output_file":"full_images/c.png","semantic_correctness":"real_train_exemplar_preserved"}
 (candidate/"metadata.jsonl").write_text(json.dumps(record)+"\n")
 coco={"images":[{"id":1,"file_name":"x.png","width":60,"height":40}],"annotations":[ann(1,1,(5,5,20,20)),ann(2,11,(35,5,15,15))],"categories":[{"id":1,"name":"common"},{"id":11,"name":"rare"}]}
 config=load_yaml(ROOT/"configs/augmentation_c.yaml");config["classes"]={"source_ids":[1],"target_ids":[11]};config["multi_room"]={"replacement_percentage":50,"random_jitter_percentage_points":0}
 manifest=build_multiroom_dataset(config,coco,{1},source,candidate,tmp_path/"out",strategy="c_multi",sample_prefix="c-copy-paste-multi")
 output=json.loads((tmp_path/"out/instances_train_augmented.json").read_text());synthetic=next(i for i in output["images"] if i.get("synthetic"))
 assert manifest["strategy"]==synthetic["strategy"]=="c_multi"
 assert synthetic["converted_annotation_ids"]==[1]
