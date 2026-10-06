"""Combine real MIO localization labels with verified five-class VisDrone data."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from data_utils import download, sha256

ROOT=Path(__file__).resolve().parents[1]
NAMES=["motorcycle","car","truck","bus","other"]
MAPPING={"motorcycle":0,"car":1,"articulated_truck":2,"pickup_truck":2,"single_unit_truck":2,"bus":3,"bicycle":4,"non-motorized_vehicle":4,"work_van":4}
SOURCE="https://tcd.miovision.com/static/dataset/"


def split_for(image_id):
    # ponytail: ID blocks reduce adjacent-frame leakage; no camera IDs are published,
    # so this is not proof of camera-disjoint evaluation. Add camera groups when available.
    bucket=int(hashlib.sha256(f"mio-block-{int(image_id)//1000}".encode()).hexdigest()[:8],16)%10
    return "test" if bucket==0 else "val" if bucket==1 else "train"


def convert_mio(rows,width,height):
    labels,ignore=[],[]
    for row in rows:
        if len(row)!=6:
            raise ValueError("Expected GT columns: image_id,class,x1,y1,x2,y2")
        category=row[1]
        x1,y1,x2,y2=map(int,row[2:])
        x1,y1=max(0,x1),max(0,y1)
        x2,y2=min(width,x2+1),min(height,y2+1)
        if x2<=x1 or y2<=y1:
            continue
        if category=="motorized_vehicle":
            # The author marks these as too small to determine the vehicle type.
            ignore.append((x1,y1,x2-1,y2-1))
        elif category in MAPPING:
            labels.append((MAPPING[category],(x1+x2)/(2*width),(y1+y2)/(2*height),(x2-x1)/width,(y2-y1)/height))
        elif category!="pedestrian":
            raise ValueError(f"Unknown MIO category {category}")
    return labels,ignore


def prepare(target):
    target=target.resolve()
    if (target/"manifest.json").exists():
        raise ValueError("Dataset already complete; retain its manifest or choose a new output.")
    target.mkdir(parents=True,exist_ok=True)
    # Prevent reuse of partially converted images with different partition logic.
    policy={"version":1,"mio_split":"sha256 id//1000 blocks 80/10/10","names":NAMES,"mapping":MAPPING}
    policy_file=target/"policy.json"
    if policy_file.exists() and json.loads(policy_file.read_text())!=policy:
        raise ValueError("Output belongs to a different preparation policy")
    policy_file.write_text(json.dumps(policy,indent=2),encoding="utf-8")
    archive=download(SOURCE+"MIO-TCD-Localization.tar",ROOT/"data/raw/MIO-TCD-Localization.tar")
    code=download(SOURCE+"MIO-TCD-Localization-Code.tar",ROOT/"data/raw/MIO-TCD-Localization-Code.tar")
    rows=defaultdict(list)
    with tarfile.open(code) as tar:
        file=tar.extractfile("MIO-TCD-Localization-Code/gt_train.csv")
        for row in csv.reader(io.TextIOWrapper(file,encoding="utf-8",newline="")):
            if row:
                rows[row[0]].append(row)
    records=[];counts=defaultdict(Counter);images=Counter();masked=Counter()

    def save_record(source,split,name,image_bytes,labels,original_sha):
        image_path=target/"images"/split/(name+".jpg")
        label_path=target/"labels"/split/(name+".txt")
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        image_path.write_bytes(image_bytes)
        text="".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for cls,x,y,w,h in labels)
        label_path.write_text(text,encoding="utf-8")
        records.append({"source":source,"split":split,"image":image_path.relative_to(target).as_posix(),"label":label_path.relative_to(target).as_posix(),
                        "original_sha256":original_sha,"image_sha256":hashlib.sha256(image_bytes).hexdigest(),"label_sha256":sha256(label_path)})
        counts[f"{source}/{split}"].update(NAMES[label[0]] for label in labels);images[f"{source}/{split}"]+=1

    total=0
    with tarfile.open(archive,mode="r|") as tar:
        for member in tar:
            match=re.fullmatch(r"MIO-TCD-Localization/train/(\d{8})\.jpg",member.name)
            if not match or not member.isfile():
                continue
            image_id=match[1];split=split_for(image_id)
            raw=tar.extractfile(member).read()
            with Image.open(io.BytesIO(raw)) as image:
                labels,ignore=convert_mio(rows.get(image_id,[]),*image.size)
                if ignore:
                    image=image.convert("RGB");draw=ImageDraw.Draw(image)
                    for box in ignore:
                        draw.rectangle(box,fill=(114,114,114))
                    buffer=io.BytesIO();image.save(buffer,format="JPEG",quality=95);data=buffer.getvalue();masked[split]+=1
                else:
                    image.verify();data=raw
            save_record("mio",split,"mio_"+image_id,data,labels,hashlib.sha256(raw).hexdigest())
            total+=1
            if total%5000==0:
                print(f"Converted {total} MIO images",flush=True)
    if total<100000:
        raise ValueError(f"Incomplete MIO training archive: {total} images")
    vis=ROOT/"data/vehicles"
    manifest=json.loads((vis/"manifest.json").read_text(encoding="utf-8"))
    if manifest["names"]!=NAMES or manifest["splits"]["train"]["images"]!=6471:
        raise ValueError("Full five-class VisDrone split required; run prepare_visdrone.py")
    for item in manifest["files"]:
        split=item["split"];name=Path(item["image"]).stem
        image=vis/"images"/split/item["image"];label=vis/"labels"/split/(name+".txt")
        if sha256(image)!=item["image_sha256"] or sha256(label)!=item["label_sha256"]:
            raise ValueError("Stale VisDrone content: "+name)
        image_path=target/"images"/split/("vis_"+name+".jpg");label_path=target/"labels"/split/("vis_"+name+".txt")
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        if not image_path.exists():
            try:
                os.link(image,image_path)
            except OSError:
                image_path.write_bytes(image.read_bytes())
        label_path.write_bytes(label.read_bytes())
        records.append({"source":"visdrone","split":split,"image":image_path.relative_to(target).as_posix(),"label":label_path.relative_to(target).as_posix(),
                        "original_sha256":item["image_sha256"],"image_sha256":item["image_sha256"],"label_sha256":item["label_sha256"]})
        images[f"visdrone/{split}"]+=1
        counts[f"visdrone/{split}"].update(NAMES[int(line.split()[0])] for line in label.read_text().splitlines() if line.strip())
    # A duplicate image may only occur in one split; held-out partitions take priority.
    priority={"test":0,"val":1,"train":2};seen=set();clean=[];duplicates=Counter()
    for record in sorted(records,key=lambda r:priority[r["split"]]):
        digest=record["original_sha256"]
        if digest in seen:
            duplicates[f"{record['source']}/{record['split']}"]+=1
            continue
        seen.add(digest);clean.append(record)
    for split in ("train","val","test"):
        paths=[str(target/r["image"]) for r in clean if r["split"]==split]
        (target/(split+".txt")).write_text("\n".join(paths)+"\n",encoding="utf-8")
        for source in ("mio","visdrone"):
            paths=[str(target/r["image"]) for r in clean if r["split"]==split and r["source"]==source]
            (target/(source+"-"+split+".txt")).write_text("\n".join(paths)+"\n",encoding="utf-8")
    # Equal source sizes for checkpoint selection; final evaluation uses full splits.
    vis_val=[r for r in clean if r["source"]=="visdrone" and r["split"]=="val"]
    mio_val=sorted((r for r in clean if r["source"]=="mio" and r["split"]=="val"),key=lambda r:hashlib.sha256(r["image"].encode()).digest())[:len(vis_val)]
    (target/"validation-balanced.txt").write_text("\n".join(str(target/r["image"]) for r in vis_val+mio_val)+"\n",encoding="utf-8")
    config={"path":target.as_posix(),"train":"train.txt","val":"validation-balanced.txt","test":"test.txt","names":dict(enumerate(NAMES))}
    (target/"data.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    for source in ("mio","visdrone"):
        config.update(train=f"{source}-train.txt",val=f"{source}-val.txt",test=f"{source}-test.txt")
        (target/(source+".yaml")).write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    summary={"policy":policy,"raw_images":dict(images),"raw_boxes":{k:dict(v) for k,v in counts.items()},"ignored_unknown_type_images":dict(masked),
             "duplicates_removed":dict(duplicates),"usable_images":dict(Counter(f"{r['source']}/{r['split']}" for r in clean)),
             "archives":[json.loads(p.with_suffix(p.suffix+".source.json").read_text()) for p in (archive,code)],
             "visdrone_manifest_sha256":sha256(vis/"manifest.json"),
             "checkpoint_validation_images":{"visdrone":len(vis_val),"mio":len(mio_val)},
             "limitations":"MIO splits use numeric ID groups, not verified camera IDs; exact duplicates removed, near-duplicate scenes may remain. VisDrone official test-dev retained. Not official MIO or VisDrone benchmark."}
    full={**summary,"files":clean}
    (target/"manifest.json").write_text(json.dumps(full,indent=2),encoding="utf-8")
    summary["manifest_sha256"]=sha256(target/"manifest.json")
    (ROOT/"reports/large-dataset.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=ROOT/"datasets/large")
    prepare(parser.parse_args().output)
