"""One-to-one event matching against manually reviewed confirmation windows."""
import argparse
import json
from pathlib import Path


def score(expected, actual, class_aware=True):
    unmatched=list(expected);matched=[];false=[]
    for prediction in sorted(actual,key=lambda p:p["confirmed_seconds"]):
        possible=[item for item in unmatched if item["zone_id"]==prediction["zone_id"] and item["min_seconds"]<=prediction["confirmed_seconds"]<=item["max_seconds"]
                  and (not class_aware or item["class"]==prediction["class"])]
        if possible:
            item=min(possible,key=lambda e:abs(prediction["confirmed_seconds"]-(e["min_seconds"]+e["max_seconds"])/2))
            unmatched.remove(item);matched.append({"expected":item,"predicted":prediction})
        else:
            false.append(prediction)
    tp,fp,fn=len(matched),len(false),len(unmatched)
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0,
            "matched":matched,"false_positives":false,"missed":unmatched}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("reference",type=Path);parser.add_argument("result",type=Path)
    args=parser.parse_args();reference=json.loads(args.reference.read_text());result=json.loads(args.result.read_text())
    if reference["source_sha256"]!=result["source_sha256"]:
        parser.error("Source hashes do not match")
    print(json.dumps({"class_aware":score(reference["events"],result["events"]),"alert_only":score(reference["events"],result["events"],False)},indent=2))
