"""Curate COCO vehicles plus camera/drone examples; keep existing held-out splits."""
import csv
import gc
import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

from data_utils import download, sha256, verify_files
from prepare_large import NAMES, MAPPING

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/"datasets/compact"
COCO_MAPPING={4:0,3:1,8:2,6:3,2:4}
COCO_HOST="https://s3.amazonaws.com/images.cocodataset.org/"


def balanced(records,limit):
    """Round-robin class coverage with deterministic ordering and unique images."""
    ordered=sorted(records,key=lambda r:hashlib.sha256(r["image"].encode()).digest())
    pools={cls:iter([r for r in ordered if cls in r["classes"]]) for cls in range(5)}
    chosen=[r for r in ordered if not r["classes"]][:min(100,limit//20)]
    seen={r["image"] for r in chosen}
    while pools and len(chosen)<limit:
        for cls in list(pools):
            item=next((r for r in pools[cls] if r["image"] not in seen),None)
            if item is None:
                del pools[cls]
            else:
                chosen.append(item);seen.add(item["image"])
                if len(chosen)==limit:
                    break
    # Include background frames if there are fewer positive images than requested.
    for item in ordered:
        if len(chosen)==limit:
            break
        if item["image"] not in seen:
            chosen.append(item);seen.add(item["image"])
    return chosen


def coco_labels(annotations,width,height):
    labels=set()
    for ann in annotations:
        if ann["category_id"] not in COCO_MAPPING:
            continue
        if ann.get("iscrowd",0):
            raise ValueError("Crowd vehicle boxes are excluded from this subset")
        x,y,w,h=ann["bbox"]
        x1,y1=max(0,x),max(0,y);x2,y2=min(width,x+w),min(height,y+h)
        if x2>x1 and y2>y1:
            labels.add((COCO_MAPPING[ann["category_id"]],(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height))
    return sorted(labels)


def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    manifest_path=TARGET/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        verify_files(TARGET,manifest["files"])
        print("Existing compact dataset verified",flush=True)
        return
    old=ROOT/"datasets/large"
    parent_hash=sha256(old/"manifest.json")
    policy={"version":1,"parent_manifest_sha256":parent_hash,"train_limits":{"mio":5000,"visdrone":2000,"coco":5000},"heldout_limit_per_source":500,"background_limit":"min(100, limit//20), if available"}
    policy_path=TARGET/"policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text())!=policy:
        raise ValueError("Partial output belongs to another selection policy")
    policy_path.write_text(json.dumps(policy,indent=2),encoding="utf-8")
    old_manifest=json.loads((old/"manifest.json").read_text())
    mio_classes=defaultdict(set)
    with tarfile.open(ROOT/"data/raw/MIO-TCD-Localization-Code.tar") as archive:
        for row in csv.reader(io.TextIOWrapper(archive.extractfile("MIO-TCD-Localization-Code/gt_train.csv"),encoding="utf-8")):
            if row and row[1] in MAPPING:
                mio_classes[row[0]].add(MAPPING[row[1]])
    candidates=defaultdict(list)
    for item in old_manifest["files"]:
        if item["source"]=="mio":
            classes=mio_classes[Path(item["image"]).stem.removeprefix("mio_")]
        else:
            classes={int(line.split()[0]) for line in (old/item["label"]).read_text().splitlines() if line.strip()}
        candidates[item["source"],item["split"]].append({**item,"classes":list(classes)})
    records=[]
    for source in ("mio","visdrone"):
        for split in ("train","val","test"):
            limit=policy["train_limits"][source] if split=="train" else 500
            selected=balanced(candidates[source,split],limit)
            if len(selected)!=limit:
                raise ValueError(f"Insufficient {source}/{split} images")
            for item in selected:
                for key in ("image","label"):
                    origin=old/item[key];destination=TARGET/item[key]
                    if sha256(origin)!=item[key+"_sha256"]:
                        raise ValueError("Modified parent data: "+item[key])
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(origin,destination)
                records.append({k:v for k,v in item.items() if k!="classes"})
            print(f"Copied {source}/{split}: {len(selected)}",flush=True)
    del old_manifest,candidates,mio_classes;gc.collect()
    annotation_zip=download(COCO_HOST+"annotations/annotations_trainval2017.zip",ROOT/"data/raw/annotations_trainval2017.zip")
    licenses=[]
    with zipfile.ZipFile(annotation_zip) as archive:
        for original_split in ("train2017","val2017"):
            with archive.open(f"annotations/instances_{original_split}.json") as file:
                document=json.load(file)
            images={item["id"]:item for item in document["images"]}
            licenses=document["licenses"]
            annotations=defaultdict(list);crowds=set()
            for ann in document["annotations"]:
                if ann["category_id"] in COCO_MAPPING:
                    if ann.get("iscrowd",0):
                        crowds.add(ann["image_id"])
                    else:
                        annotations[ann["image_id"]].append({"category_id":ann["category_id"],"bbox":ann["bbox"]})
            del document;gc.collect()
            options=[]
            for image_id,image in images.items():
                if image_id in crowds:
                    continue
                anns=annotations.get(image_id,[])
                if original_split=="train2017" and not anns:
                    continue
                if image["file_name"]!=f"{image_id:012d}.jpg":
                    raise ValueError("Unexpected COCO filename")
                options.append({"image":f"coco_{image_id:012d}.jpg","info":image,"annotations":anns,"classes":list({COCO_MAPPING[a["category_id"]] for a in anns})})
            if original_split=="train2017":
                groups=[("train",balanced(options,5000))]
            else:
                # Official COCO validation only; separate halves for selection and final test.
                val_pool=[r for r in options if r["info"]["id"]%2==0]
                test_pool=[r for r in options if r["info"]["id"]%2==1]
                groups=[("val",balanced(val_pool,500)),("test",balanced(test_pool,500))]
            for split,selected in groups:
                expected=5000 if split=="train" else 500
                if len(selected)!=expected:
                    raise ValueError(f"Insufficient COCO/{split} vehicle images")
                def save(item):
                    image_path=TARGET/"images"/split/item["image"]
                    label_path=TARGET/"labels"/split/(Path(item["image"]).stem+".txt")
                    image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
                    url=COCO_HOST+original_split+"/"+item["info"]["file_name"]
                    download(url,image_path,quiet=True)
                    with Image.open(image_path) as image:
                        if image.size!=(item["info"]["width"],item["info"]["height"]):
                            raise ValueError("COCO image dimensions differ from GT")
                        labels=coco_labels(item["annotations"],*image.size);image.verify()
                    label_path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c,x,y,w,h in labels),encoding="utf-8")
                    return {"source":"coco","split":split,"image":image_path.relative_to(TARGET).as_posix(),"label":label_path.relative_to(TARGET).as_posix(),"original_sha256":sha256(image_path),"image_sha256":sha256(image_path),"label_sha256":sha256(label_path),"source_url":url,"license_id":item["info"]["license"]}
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for start in range(0,len(selected),250):
                        records.extend(pool.map(save,selected[start:start+250]))
                        print(f"Downloaded COCO/{split}: {min(start+250,len(selected))}/{len(selected)}",flush=True)
            del options,images,annotations,groups;gc.collect()
    priority={"test":0,"val":1,"train":2};seen=set();clean=[];removed=Counter()
    for item in sorted(records,key=lambda r:priority[r["split"]]):
        if item["original_sha256"] in seen:
            removed[f"{item['source']}/{item['split']}"]+=1
            continue
        seen.add(item["original_sha256"]);clean.append(item)
    counts=Counter(f"{r['source']}/{r['split']}" for r in clean);boxes=defaultdict(Counter)
    for item in clean:
        for line in (TARGET/item["label"]).read_text().splitlines():
            if line.strip():
                boxes[f"{item['source']}/{item['split']}"].update([NAMES[int(line.split()[0])]])
    sources=["mio","visdrone","coco"]
    for split in ("train","val","test"):
        (TARGET/(split+".txt")).write_text("\n".join(str(TARGET/r["image"]) for r in clean if r["split"]==split)+"\n",encoding="utf-8")
        for source in sources:
            (TARGET/f"{source}-{split}.txt").write_text("\n".join(str(TARGET/r["image"]) for r in clean if r["split"]==split and r["source"]==source)+"\n",encoding="utf-8")
    config={"path":TARGET.as_posix(),"train":"train.txt","val":"val.txt","test":"test.txt","names":dict(enumerate(NAMES))}
    (TARGET/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    for source in sources:
        (TARGET/(source+".yaml")).write_text(yaml.safe_dump({**config,"train":f"{source}-train.txt","val":f"{source}-val.txt","test":f"{source}-test.txt"},sort_keys=False),encoding="utf-8")
    summary={"dataset":"Compact traffic + COCO vehicles","sources":sources,"names":NAMES,"policy":policy,"usable_images":dict(counts),"boxes":{k:dict(v) for k,v in boxes.items()},"duplicates_removed":dict(removed),"coco_annotations":json.loads(annotation_zip.with_suffix(".zip.source.json").read_text()),"coco_mapping":COCO_MAPPING,"coco_licenses":licenses,"limitations":"Curated, class-covered subset; not official dataset benchmark. COCO val halves were not fine-tuned on, but COCO pretrained model development used COCO. MIO camera-disjointness is unverified; near duplicates may remain. Images retain original licenses; MIO noncommercial and VisDrone author terms apply."}
    verify_files(TARGET,clean)
    manifest_path.write_text(json.dumps({**summary,"files":clean},indent=2),encoding="utf-8")
    summary["manifest_sha256"]=sha256(manifest_path)
    (ROOT/"reports/compact-dataset.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
