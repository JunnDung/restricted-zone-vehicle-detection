"""One-to-one event matching against manually reviewed confirmation windows."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from zones import ZoneMap


def score(expected, actual, class_aware=True, *, duration_seconds=None):
    if duration_seconds is not None:
        if not math.isfinite(duration_seconds) or duration_seconds<=0:
            raise ValueError('Observed duration must be finite and positive')
        if any(not math.isfinite(p['confirmed_seconds']) or not 0<=p['confirmed_seconds']<duration_seconds for p in actual):
            raise ValueError('Alert timestamps must lie inside the observed video')
    unmatched=list(expected);matched=[];false=[]
    for prediction in sorted(actual,key=lambda p:p["confirmed_seconds"]):
        possible=[item for item in unmatched if item["zone_id"]==prediction["zone_id"] and item["min_seconds"]<=prediction["confirmed_seconds"]<=item["max_seconds"]
                  and (not class_aware or item["class"]==prediction["class"])]
        if possible:
            # Preserve later matches when confirmation windows overlap.
            item=min(possible,key=lambda e:e['max_seconds'])
            unmatched.remove(item);matched.append({"expected":item,"predicted":prediction})
        else:
            false.append(prediction)
    tp,fp,fn=len(matched),len(false),len(unmatched)
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0,
            "observed_seconds":duration_seconds,"false_positives_per_hour":fp*3600/duration_seconds if duration_seconds is not None else None,
            "matched":matched,"false_positives":false,"missed":unmatched}


def evaluate(reference,result,zone_map):
    if reference['source_sha256']!=result['source_sha256']:
        raise ValueError('Source hashes do not match')
    if ZoneMap.model_validate(result['settings']['zone_map'])!=ZoneMap.model_validate(zone_map):
        raise ValueError('Zone maps do not match the reference')
    metadata=result['metadata'];frames=result['processed_frames'];fps=metadata['fps']
    if metadata.get('live') or not math.isfinite(fps) or fps<=0:
        raise ValueError('A finite positive video FPS is required')
    if result['phase']!='completed' or frames<=0 or frames!=metadata['frames']:
        raise ValueError('Evaluate the full reference video; partial or stopped runs are not supported')
    duration=frames/fps
    if any(not 0<=e['min_seconds']<=e['max_seconds']<=duration for e in reference['events']):
        raise ValueError('Reference windows must lie inside the observed video')
    return {'class_aware':score(reference['events'],result['events'],duration_seconds=duration),
            'alert_only':score(reference['events'],result['events'],False,duration_seconds=duration)}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("reference",type=Path);parser.add_argument("result",type=Path)
    parser.add_argument('--output',type=Path,help='Save the evaluation as UTF-8 JSON')
    args=parser.parse_args();reference=json.loads(args.reference.read_text(encoding='utf-8'));result=json.loads(args.result.read_text(encoding='utf-8'))
    zone_map=ZoneMap.model_validate_json((ROOT/reference['zone_map']).read_text(encoding='utf-8'))
    try:
        report=evaluate(reference,result,zone_map)
    except ValueError as exc:
        parser.error(str(exc))
    text=json.dumps(report,indent=2,allow_nan=False)
    if args.output:
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
