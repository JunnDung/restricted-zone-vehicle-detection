"""Train on local CUDA, evaluate held-out domains, promote only after validation."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["YOLO_VERBOSE"]="False"
import torch
from ultralytics import YOLO

from data_utils import sha256

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import atomic_json, analyze
from zones import ZoneMap
from evaluate_events import score


def metrics_dict(metrics):
    return {"precision":float(metrics.box.mp),"recall":float(metrics.box.mr),"mAP50":float(metrics.box.map50),"mAP50_95":float(metrics.box.map),
            "per_class":[{"class":metrics.names[int(cls)],"AP50":float(metrics.box.ap50[i]),"AP50_95":float(metrics.box.ap[i])} for i,cls in enumerate(metrics.box.ap_class_index)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"datasets/large")
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--imgsz",type=int,choices=[640,960],default=640)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--backup-github",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Local CUDA GPU required; install setup.ps1 -Cuda")
    if not 1<=args.epochs<=200 or not 1<=args.batch<=32:
        parser.error("epochs 1–200, batch 1–32")
    data=args.data.resolve();run=ROOT/"runs/zoneguard_large"
    if run.exists() and not args.resume:
        parser.error("Run already exists. Use --resume to retain checkpoints.")
    if not (data/"manifest.json").exists():
        parser.error("Run scripts/prepare_large.py first")
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    manifest_hash=sha256(data/"manifest.json")
    run_settings={"dataset_manifest_sha256":manifest_hash,"epochs":args.epochs,"batch":args.batch,"imgsz":args.imgsz,"seed":42}
    saved=ROOT/"reports/large-run-settings.json"
    if args.resume:
        old=json.loads(saved.read_text())
        if old["dataset_manifest_sha256"]!=manifest_hash or old["imgsz"]!=args.imgsz:
            parser.error("Resume dataset/image size must match the original run")
    else:
        atomic_json(saved,run_settings)
    torch.set_num_threads(4)
    live=ROOT/"reports/training-live.json"
    state={"status":"verifying","started_utc":datetime.now(timezone.utc).isoformat(),"device":torch.cuda.get_device_name(0),"epochs":args.epochs,"epoch":0,
           "dataset_manifest_sha256":manifest_hash,"train_images":sum(n for k,n in manifest["usable_images"].items() if k.endswith("/train"))}
    atomic_json(live,state)
    # Verify once before optimization. A changed image invalidates the experiment.
    for i,item in enumerate(manifest["files"]):
        if sha256(data/item["image"])!=item["image_sha256"] or sha256(data/item["label"])!=item["label_sha256"]:
            raise ValueError("Stale dataset content: "+item["image"])
        if i%10000==0:
            print(f"Verified {i}/{len(manifest['files'])} files",flush=True)
    # A second project can occupy the same laptop GPU; keep its process intact.
    while torch.cuda.mem_get_info()[0] < 4*1024**3:
        state.update(status="waiting_gpu",free_vram_mb=round(torch.cuda.mem_get_info()[0]/1024**2))
        atomic_json(live,state)
        time.sleep(10)
    state["status"]="evaluating";atomic_json(live,state)
    baseline_report=ROOT/"reports/large-baseline-validation.json"
    def evaluate(weights,source,split,name):
        return metrics_dict(YOLO(str(weights)).val(data=str(data/(source+".yaml")),split=split,imgsz=args.imgsz,batch=args.batch,device="0",workers=0,
                                                project=str(ROOT/"runs"),name=name,exist_ok=True,plots=False,verbose=False))
    if not args.resume:
        baseline={source:evaluate(ROOT/"models/vehicle-yolo11n.pt",source,"val","baseline_"+source+"_val") for source in ("mio","visdrone")}
        atomic_json(baseline_report,{"dataset_manifest_sha256":manifest_hash,"metrics":baseline})
    else:
        baseline=json.loads(baseline_report.read_text())["metrics"]
    state["status"]="training";atomic_json(live,state)
    def epoch_end(trainer):
        state.update(epoch=trainer.epoch+1,epochs=trainer.epochs,status="training",metrics={k:float(v) for k,v in trainer.metrics.items()},updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(live,state)
        print(json.dumps({"epoch":state["epoch"],"epochs":state["epochs"],"metrics":state["metrics"]}),flush=True)
    model=YOLO(str(run/"weights/last.pt") if args.resume else str(ROOT/"models/yolo11s.pt"))
    model.add_callback("on_fit_epoch_end",epoch_end)
    started=time.monotonic()
    if args.resume:
        model.train(resume=True,device="0",workers=0)
    else:
        model.train(data=str(data/"data.yaml"),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,device="0",workers=0,seed=42,deterministic=True,
                    project=str(ROOT/"runs"),name="zoneguard_large",exist_ok=True,patience=8,cache=False,amp=True,
                    mosaic=.5,close_mosaic=5,scale=.3,save_period=5)
    best=run/"weights/best.pt"
    state["status"]="evaluating";atomic_json(live,state)
    validation={source:evaluate(best,source,"val","large_"+source+"_val") for source in ("mio","visdrone")}
    old_mean=sum(m["mAP50_95"] for m in baseline.values())/2
    new_mean=sum(m["mAP50_95"] for m in validation.values())/2
    # Promotion is decided on validation before looking at either test split.
    accepted=new_mean>old_mean+.005 and validation["visdrone"]["mAP50_95"]>=baseline["visdrone"]["mAP50_95"]-.02
    reference=json.loads((ROOT/"reports/parking-reference.json").read_text())
    video_scores={}
    for key,weights in (("baseline",ROOT/"models/vehicle-yolo11n.pt"),("candidate",best)):
        output=ROOT/".local"/("admission-"+key+"-"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        result=analyze(ROOT/reference["source"],output,ZoneMap.model_validate_json((ROOT/reference["zone_map"]).read_text()),weights=weights,device="0",imgsz=args.imgsz)
        if result["source_sha256"]!=reference["source_sha256"]:
            raise ValueError("Admission video was modified")
        video_scores[key]={"class_aware":score(reference["events"],result["events"]),"alert_only":score(reference["events"],result["events"],False)}
    accepted=accepted and all(video_scores["candidate"][metric]["f1"]>=video_scores["baseline"][metric]["f1"] for metric in ("class_aware","alert_only"))
    test={source:evaluate(best,source,"test","large_"+source+"_test") for source in ("mio","visdrone")}
    report={"architecture":"YOLO11s","trained_locally":True,"device":torch.cuda.get_device_name(0),"settings":run_settings,
            "actual_epochs":state["epoch"],"elapsed_training_seconds":round(time.monotonic()-started,1),"dataset_manifest_sha256":manifest_hash,
            "base_weights_sha256":sha256(ROOT/"models/yolo11s.pt"),"weights_sha256":sha256(best),"baseline_validation":baseline,
            "validation":validation,"test":test,"admission_video":video_scores,"accepted":accepted,"balanced_validation_mean_before":old_mean,"balanced_validation_mean_after":new_mean,
            "promotion_rule":"mean val mAP50-95 improves >0.005; VisDrone val does not fall more than 0.02; neither alert nor class-aware F1 regresses on the admission video. Selection made before test evaluation.",
            "limitations":manifest["limitations"]+" Detector metrics do not measure zone violations. Camera-specific video verification is still required."}
    atomic_json(ROOT/"reports/large-training.json",report)
    shutil.copy2(run/"results.csv",ROOT/"reports/large-training-curve.csv")
    if accepted:
        candidate=ROOT/"models/vehicle-large.pt"
        temporary=candidate.with_suffix(".tmp")
        shutil.copy2(best,temporary);os.replace(temporary,candidate)
    state.update(status="completed",accepted=accepted,report="reports/large-training.json",finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(live,state)
    if args.backup_github:
        paths=["reports/large-training.json","reports/large-training-curve.csv","reports/large-baseline-validation.json","reports/large-run-settings.json"]
        if accepted:
            paths.append("models/vehicle-large.pt")
        subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
        subprocess.run(["git","commit","-m","model: train and evaluate YOLO11s on expanded traffic datasets"],cwd=ROOT,check=True)
        subprocess.run(["git","push"],cwd=ROOT,check=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        path=ROOT/"reports/training-live.json"
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status="error",error=str(exc),updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path,state)
        raise
