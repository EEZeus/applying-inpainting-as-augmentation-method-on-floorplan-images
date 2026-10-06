"""Compose semantically verified rare-exemplar LoRA edits into multi-room plans."""
import copy, math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .pipeline import _atomic_json,_copy_original_train_images,file_digest,latest_metadata,read_metadata,validate_augmented_coco

def pair_key(record):
    return (int(record["target_annotation_id"]), int(record["donor_annotation_id"]))


def build_multiroom_dataset(config,coco,train_ids,image_dir,candidate_dir,output_dir,
                            strategy="lora_multi", sample_prefix="lora-multi-exemplar",
                            allowed_pairs=None, canonical_order=False):
    """Compose accepted single-room edits into multi-room plans.

    ``allowed_pairs`` restricts the pool to these (target, donor) annotation
    pairs. ``canonical_order`` sorts each plan's pool by pair before the seeded
    shuffle, so two candidate directories holding the same pairs (e.g. copy-paste
    and inpainting of identical pairings) are composed into identical groups.
    Defaults reproduce the original behaviour exactly.
    """
    if strategy not in {"lora_multi", "c_multi"}:
        raise ValueError(f"Unsupported multi-room strategy: {strategy}")
    fraction=float(config["multi_room"]["replacement_percentage"])/100;jitter=float(config["multi_room"]["random_jitter_percentage_points"])/100
    if not 0<fraction<=1 or jitter<0 or fraction-jitter<=0 or fraction+jitter>1:raise ValueError("percentage ± jitter must remain within (0,100]")
    candidate_dir,output_dir=Path(candidate_dir),Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    records=[r for r in latest_metadata(read_metadata(candidate_dir/"metadata.jsonl")).values() if r["status"]=="accepted"]
    if allowed_pairs is not None:
        allowed=set(tuple(map(int,pair)) for pair in allowed_pairs);records=[r for r in records if pair_key(r) in allowed]
        if {pair_key(r) for r in records}!=allowed:raise ValueError("Some allowed pairs are not accepted in this candidate directory")
    common=set(map(int,config["classes"]["source_ids"]));rare=set(map(int,config["classes"]["target_ids"]))
    for r in records:
        if int(r["source_category_id"]) not in common or int(r["category_id"]) not in rare or int(r.get("donor_category_id",-1)) not in rare:
            raise ValueError("Every composition must be common source -> genuine rare donor/target")
        if r.get("semantic_correctness")!="real_train_exemplar_preserved":raise ValueError("Candidate lacks preserved real rare semantics")
    grouped=defaultdict(list)
    for r in records:grouped[int(r["source_image_id"])].append(r)
    images={int(i["id"]):i for i in coco["images"]};anns=defaultdict(list)
    for a in coco["annotations"]:anns[int(a["image_id"])].append(a)
    rng=np.random.default_rng(int(config["seed"]));max_im=max(i["id"] for i in coco["images"])+9_000_000;max_ann=max(a["id"] for a in coco["annotations"])+9_000_000_000;stride=max(Counter(a["image_id"] for a in coco["annotations"]).values())
    result={k:copy.deepcopy(v) for k,v in coco.items() if k not in {"images","annotations"}};result["images"]=_copy_original_train_images(coco,train_ids,image_dir,output_dir);result["annotations"]=[copy.deepcopy(a) for a in coco["annotations"] if int(a["image_id"]) in train_ids]
    groups=[];index=0
    for source_id in sorted(grouped):
        pool=grouped[source_id]
        if canonical_order:pool.sort(key=pair_key)
        rng.shuffle(pool);common_count=sum(int(a["category_id"]) in common for a in anns[source_id]);cursor=0
        while cursor<len(pool):
            sampled=float(rng.uniform(fraction-jitter,fraction+jitter)) if jitter else fraction;desired=max(1,int(math.floor(sampled*common_count)+(rng.random()<sampled*common_count%1)))
            batch=pool[cursor:cursor+desired];cursor+=len(batch);info=images[source_id];original=np.asarray(Image.open(Path(image_dir)/info["file_name"]).convert("RGB"));combined=original.copy();union=np.zeros(original.shape[:2],bool);relabel={};conversions=[]
            for r in batch:
                candidate=np.asarray(Image.open(candidate_dir/r["output_file"]).convert("RGB"));changed=np.any(candidate!=original,axis=2)
                if np.any(union&changed):raise ValueError("Rare-room edit regions overlap")
                combined[changed]=candidate[changed];union|=changed;relabel[int(r["target_annotation_id"])]=int(r["category_id"]);conversions.append({"source_annotation_id":int(r["target_annotation_id"]),"source_category_id":int(r["source_category_id"]),"target_category_id":int(r["category_id"]),"donor_annotation_id":int(r["donor_annotation_id"]),"donor_category_id":int(r["donor_category_id"])})
            if not np.array_equal(combined[~union],original[~union]):raise ValueError("Outside pixels changed")
            sample=f"{sample_prefix}-im{source_id}-g{index}";path=output_dir/"full_images"/f"{sample}.png";path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(combined).save(path);synthetic_id=max_im+index;start=max_ann+index*stride
            out=copy.deepcopy(info);out.update(id=synthetic_id,file_name=f"full_images/{path.name}",synthetic=True,strategy=strategy,source_image_id=source_id,sample_id=sample,converted_annotation_ids=sorted(relabel),replacement_count=len(batch),common_room_count=common_count)
            result["images"].append(out)
            for pos,a in enumerate(sorted(anns[source_id],key=lambda x:int(x["id"]))):
                duplicate=copy.deepcopy(a);duplicate.update(id=start+pos,image_id=synthetic_id)
                if int(a["id"]) in relabel:duplicate["category_id"]=relabel[int(a["id"])]
                result["annotations"].append(duplicate)
            groups.append({"sample_id":sample,"source_image_id":source_id,"conversions":conversions,"replacement_count":len(batch),"common_room_count":common_count,"requested_replacement_fraction":sampled,"achieved_replacement_fraction":len(batch)/common_count if common_count else 0,"candidate_pool_limited":len(batch)<desired,"outside_change_ratio":0.,"boundary_change_ratio":0.,"output_sha256":file_digest(path),"pair_keys":[list(pair_key(r)) for r in batch]});index+=1
    result["images"].sort(key=lambda x:int(x["id"]));result["annotations"].sort(key=lambda x:int(x["id"]));validate_augmented_coco(result,output_dir,train_ids);_atomic_json(output_dir/"instances_train_augmented.json",result)
    manifest={"strategy":strategy,"seed":int(config["seed"]),"replacement_fraction":fraction,"fraction_jitter":jitter,"source_candidate_count":len(records),"synthetic_plans":len(groups),"groups":groups,"all_sources_are_common":True,"all_targets_and_donors_are_rare":True,"semantic_correctness":"real_train_exemplar_preserved","max_outside_change_ratio":0.,"max_boundary_change_ratio":0.};_atomic_json(output_dir/"multiroom_manifest.json",manifest);return manifest
