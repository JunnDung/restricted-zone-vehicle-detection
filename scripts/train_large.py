"""Train on local CUDA, evaluate held-out domains, promote only after validation."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

os.environ["YOLO_VERBOSE"]="False"
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import strip_optimizer

from data_utils import sha256, verify_files, memory_headroom_mb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline import atomic_json, analyze
from zones import ZoneMap
from evaluate_events import score


def metrics_dict(metrics):
    return {"precision":float(metrics.box.mp),"recall":float(metrics.box.mr),"mAP50":float(metrics.box.map50),"mAP50_95":float(metrics.box.map),
            "speed_ms_per_image":{key:float(value) for key,value in metrics.speed.items()},
            "per_class":[{"class":metrics.names[int(cls)],"AP50":float(metrics.box.ap50[i]),"AP50_95":float(metrics.box.ap[i])} for i,cls in enumerate(metrics.box.ap_class_index)]}


def detector_admission(baseline,candidate):
    old=sum(m['mAP50_95'] for m in baseline.values())/len(baseline)
    new=sum(candidate[s]['mAP50_95'] for s in baseline)/len(baseline)
    return new>old+.005 and all(candidate[s]['mAP50_95']>=baseline[s]['mAP50_95']-.02 for s in baseline),old,new


def source_replay(records):
    groups={source:sorted((r['image'] for r in records if r['split']=='train' and r['source']==source),key=lambda p:hashlib.sha256(p.encode()).digest()) for source in ('mio','visdrone','coco')}
    if not all(groups.values()):
        raise ValueError('All three training sources are required for replay')
    count=max(map(len,groups.values()))
    return [images[i%len(images)] for images in groups.values() for i in range(count)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"datasets/compact")
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--workers",type=int,choices=[0,2],default=0)
    parser.add_argument("--imgsz",type=int,choices=[640,960],default=640)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--replay",action="store_true",help="Short, validation-driven refinement with equal source sampling")
    parser.add_argument("--backup-github",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Local CUDA GPU required; install setup.ps1 -Cuda")
    environment=os.environ.get('ZONEGUARD_TRAINING_ENV','local')
    if not 1<=args.epochs<=200 or not 1<=args.batch<=32:
        parser.error("epochs 1–200, batch 1–32")
    data=args.data.resolve();run=ROOT/'runs'/('zoneguard_replay' if args.replay else 'zoneguard_compact')
    if run.exists() and not args.resume:
        parser.error("Run already exists. Use --resume to retain checkpoints.")
    if args.resume and not (run/"weights/last.pt").exists():
        parser.error("No completed epoch checkpoint to resume")
    if not (data/"manifest.json").exists():
        parser.error("Run scripts/prepare_compact.py first")
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    manifest_hash=sha256(data/"manifest.json")
    sources=manifest["sources"]
    limitations=manifest["limitations"]
    base_weights=ROOT/"models/vehicle-yolo11n.pt"
    initial_weights=ROOT/'runs/zoneguard_compact/weights/deployment.pt' if args.replay else base_weights
    training_yaml=data/'data.yaml'
    sampling={'policy':'original compact selection','unique_train_images':sum(n for k,n in manifest['usable_images'].items() if k.endswith('/train'))}
    if args.replay:
        paths=source_replay(manifest['files'])
        index=data/'replay-train.txt';index.write_text('\n'.join(str(data/path) for path in paths)+'\n')
        config=yaml.safe_load(training_yaml.read_text());config['train']=index.name
        training_yaml=data/'replay.yaml';training_yaml.write_text(yaml.safe_dump(config,sort_keys=False))
        sampling.update(policy='equal source sampling; deterministic repetition of VisDrone train images only',samples=len(paths),index_sha256=sha256(index))
    run_settings={"dataset_manifest_sha256":manifest_hash,"epochs":args.epochs,"batch":args.batch,"imgsz":args.imgsz,"seed":42,
                  "architecture":"YOLO11n","base_weights_sha256":sha256(base_weights),"initial_weights_sha256":sha256(initial_weights),"sampling":sampling,
                  "mosaic":0,"optimizer":"SGD","lr0":.0005 if args.replay else .002,"warmup_epochs":.5 if args.replay else 3.,
                  "warmup_bias_lr":.0005 if args.replay else .1,"scale":.1 if args.replay else .3,"workers":args.workers,"cache":False,"amp":True}
    saved=ROOT/'reports'/('replay-run-settings.json' if args.replay else 'compact-run-settings.json')
    if args.resume:
        old=json.loads(saved.read_text())
        if any(old.get(k)!=run_settings[k] for k in ("dataset_manifest_sha256","imgsz","architecture","base_weights_sha256")):
            parser.error("Resume dataset, model and image size must match the original run")
        run_settings={**old,"batch":args.batch,"workers":args.workers,"execution_environment":environment}
        atomic_json(saved,run_settings)
    else:
        atomic_json(saved,run_settings)
    torch.set_num_threads(4)
    live=ROOT/"reports/training-live.json"
    state={"status":"verifying","started_utc":datetime.now(timezone.utc).isoformat(),"device":torch.cuda.get_device_name(0),"epochs":args.epochs,"epoch":0,
           "execution_environment":environment,"training_phase":'source_replay' if args.replay else 'compact',"dataset_manifest_sha256":manifest_hash,"train_images":sum(n for k,n in manifest["usable_images"].items() if k.endswith("/train"))}
    atomic_json(live,state)
    last_update=0.0
    def publish(force=False):
        nonlocal last_update
        if force or time.monotonic()-last_update>=15:
            last_update=time.monotonic()
            state["updated_utc"]=datetime.now(timezone.utc).isoformat()
            atomic_json(live,state)
            print(json.dumps(state),flush=True)
    def memory_ready():
        previous=state["status"];waited=False
        while (free:=memory_headroom_mb()) is not None and free<2048:
            waited=True;state.update(status="waiting_memory",memory_headroom_mb=free)
            publish(True);time.sleep(10)
        if waited:
            state.update(status=previous,memory_headroom_mb=free);publish(True)
    # Verify once before optimization. A changed image invalidates the experiment.
    def verified(done,total):
        state.update(verified_images=done,dataset_images=total)
        publish(True)
    verify_files(data,manifest["files"],verified)
    del manifest;gc.collect()
    memory_ready()
    # A second project can occupy the same laptop GPU; keep its process intact.
    while torch.cuda.mem_get_info()[0] < 2.5*1024**3:
        state.update(status="waiting_gpu",free_vram_mb=round(torch.cuda.mem_get_info()[0]/1024**2))
        atomic_json(live,state)
        time.sleep(10)
    state["status"]="evaluating";atomic_json(live,state)
    baseline_report=ROOT/"reports"/("colab-baseline-validation.json" if environment=='google_colab' else "compact-baseline-validation.json")
    def validation_callbacks(model,source,split):
        def start(validator):
            state.update(status="evaluating",evaluation_source=source,evaluation_split=split,batch=0,batches=len(validator.dataloader))
            publish(True)
        def batch(validator):
            state["batch"]=validator.batch_i+1
            publish(state["batch"]==1)
            memory_ready()
        model.add_callback("on_val_start",start)
        model.add_callback("on_val_batch_end",batch)
    def evaluate(weights,source,split,name):
        memory_ready()
        model=YOLO(str(weights))
        validation_callbacks(model,source,split)
        try:
            return metrics_dict(model.val(data=str(data/(source+".yaml")),split=split,imgsz=args.imgsz,batch=args.batch,device="0",workers=args.workers,
                                          project=str(ROOT/"runs"),name=name,exist_ok=True,plots=False,verbose=False))
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    if not args.resume or not baseline_report.exists():
        baseline={source:evaluate(base_weights,source,"val","compact_baseline_"+source+"_val") for source in sources}
        atomic_json(baseline_report,{"dataset_manifest_sha256":manifest_hash,"metrics":baseline})
    else:
        baseline=json.loads(baseline_report.read_text())["metrics"]
    state.update(status="preparing_train",batch=0,batches=0);publish(True)
    def epoch_start(trainer):
        state.update(status="training",current_epoch=trainer.epoch+1,batch=0,batches=len(trainer.train_loader))
        publish(True)
        memory_ready()
    def batch_end(trainer):
        state["batch"]+=1
        publish(state["batch"]==1)
        memory_ready()
    def epoch_end(trainer):
        state.update(epoch=trainer.epoch+1,epochs=trainer.epochs,status="training",metrics={k:float(v) for k,v in trainer.metrics.items()},updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(live,state)
        print(json.dumps({"epoch":state["epoch"],"epochs":state["epochs"],"metrics":state["metrics"]}),flush=True)
        if environment=='google_colab' and state['epoch']>state.get('balanced_evaluated_epoch',0):
            validation={source:evaluate(trainer.last,source,'val','selection_'+source) for source in sources}
            eligible,_,mean=detector_admission(baseline,validation)
            selected=ROOT/'reports'/('colab-replay-checkpoint-selection.json' if args.replay else 'colab-checkpoint-selection.json')
            previous=json.loads(selected.read_text()) if selected.exists() else {'mean':-1}
            if eligible and mean>previous['mean']:
                shutil.copy2(trainer.last,run/'weights/balanced-best.pt')
                atomic_json(selected,{'epoch':state['epoch'],'mean':mean,'validation':validation,'dataset_manifest_sha256':manifest_hash})
            state.update(status='training',balanced_evaluated_epoch=state['epoch'])
            publish(True)
    memory_ready()
    model=YOLO(str(run/"weights/last.pt") if args.resume else str(initial_weights))
    restored_epochs=int(model.ckpt.get('epoch',-1))+1 if args.resume else 0
    state['epoch']=restored_epochs
    model.add_callback("on_train_epoch_start",epoch_start)
    model.add_callback("on_train_batch_end",batch_end)
    model.add_callback("on_fit_epoch_end",epoch_end)
    validation_callbacks(model,"three_sources","val")
    started=time.monotonic()
    if args.resume:
        model.train(resume=True,device="0",batch=args.batch,workers=args.workers)
    else:
        model.train(data=str(training_yaml),epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,device="0",workers=args.workers,seed=42,deterministic=True,
                    project=str(ROOT/"runs"),name=run.name,exist_ok=True,patience=5,cache=False,amp=True,
                    mosaic=0,close_mosaic=0,scale=run_settings['scale'],save_period=5,plots=False,optimizer="SGD",lr0=run_settings['lr0'],
                    warmup_epochs=run_settings['warmup_epochs'],warmup_bias_lr=run_settings['warmup_bias_lr'])
    training_seconds=time.monotonic()-started
    best=run/'weights/balanced-best.pt' if (run/'weights/balanced-best.pt').exists() else run/"weights/best.pt"
    parameters=sum(parameter.numel() for parameter in model.model.parameters())
    deployment=run/'weights/deployment.pt'
    strip_optimizer(str(best),s=str(deployment))
    best=deployment
    del model;gc.collect();torch.cuda.empty_cache()
    state["status"]="evaluating";atomic_json(live,state)
    validation={source:evaluate(best,source,"val","compact_"+source+"_val") for source in sources}
    # Promotion is decided on validation before looking at either test split.
    accepted,old_mean,new_mean=detector_admission(baseline,validation)
    reference=json.loads((ROOT/"reports/parking-reference.json").read_text())
    video_scores={}
    for key,weights in (("baseline",ROOT/"models/vehicle-yolo11n.pt"),("candidate",best)):
        output=ROOT/".local"/("admission-"+key+"-"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        result=analyze(ROOT/reference["source"],output,ZoneMap.model_validate_json((ROOT/reference["zone_map"]).read_text()),weights=weights,device="0",imgsz=args.imgsz)
        if result["source_sha256"]!=reference["source_sha256"]:
            raise ValueError("Admission video was modified")
        video_scores[key]={"class_aware":score(reference["events"],result["events"]),"alert_only":score(reference["events"],result["events"],False),
                           "processing_fps":result['processing_fps'],"inference_ms":result['inference_ms'],"processed_frames":result['processed_frames'],
                           "elapsed_seconds":result['elapsed_seconds'],"includes_video_encoding":True}
        gc.collect();torch.cuda.empty_cache()
    accepted=accepted and all(video_scores["candidate"][metric]["f1"]>=video_scores["baseline"][metric]["f1"] for metric in ("class_aware","alert_only"))
    baseline_test={source:evaluate(base_weights,source,"test","compact_baseline_"+source+"_test") for source in sources}
    test={source:evaluate(best,source,"test","compact_"+source+"_test") for source in sources}
    report={"architecture":"YOLO11n","trained_locally":environment=='local',"execution_environment":environment,"restored_epochs":restored_epochs,
            "torch_version":torch.__version__,"parameters":parameters,"weights_bytes":best.stat().st_size,"device":torch.cuda.get_device_name(0),"settings":run_settings,
            "actual_epochs":state["epoch"],"elapsed_training_seconds":round(training_seconds,1),"elapsed_training_and_evaluation_seconds":round(time.monotonic()-started,1),"dataset_manifest_sha256":manifest_hash,
            "base_weights_sha256":sha256(base_weights),"weights_sha256":sha256(best),"baseline_validation":baseline,
            "validation":validation,"baseline_test":baseline_test,"test":test,"admission_video":video_scores,"accepted":accepted,"balanced_validation_mean_before":old_mean,"balanced_validation_mean_after":new_mean,
            "promotion_rule":"mean val mAP50-95 improves >0.005; no source val falls more than 0.02; neither alert nor class-aware F1 regresses on the admission video. Selection made before test evaluation.",
            "limitations":limitations+" Detector metrics do not measure zone violations. Camera-specific video verification is still required."}
    report_name='replay-training' if args.replay else 'compact-training'
    atomic_json(ROOT/'reports'/(report_name+'.json'),report)
    shutil.copy2(run/"results.csv",ROOT/'reports'/(report_name+'-curve.csv'))
    if accepted:
        candidate=ROOT/"models/vehicle-large.pt"
        temporary=candidate.with_suffix(".tmp")
        shutil.copy2(best,temporary);os.replace(temporary,candidate)
    state.update(status="completed",accepted=accepted,report='reports/'+report_name+'.json',finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(live,state)
    if args.backup_github:
        paths=['reports/'+report_name+'.json','reports/'+report_name+'-curve.csv',str(baseline_report.relative_to(ROOT)),str(saved.relative_to(ROOT))]
        if accepted:
            paths.append("models/vehicle-large.pt")
        subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
        subprocess.run(["git","commit","-m","model: evaluate compact YOLO11n traffic fine-tuning"],cwd=ROOT,check=True)
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
