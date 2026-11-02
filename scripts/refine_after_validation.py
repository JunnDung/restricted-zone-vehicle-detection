"""Run the predeclared three-epoch source-replay trial only if validation needs it."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    deadline=time.monotonic()+4*3600
    status=ROOT/'reports/colab-bootstrap.json'
    while time.monotonic()<deadline:
        phase=json.loads(status.read_text())['phase'] if status.exists() else ''
        if phase=='error':
            raise RuntimeError('Initial training failed; inspect the original log before refinement')
        if phase=='completed':
            break
        time.sleep(10)
    else:
        raise TimeoutError('Initial training has not completed')
    # Only validation enters the refinement decision; test fields are not consulted.
    from train_large import detector_admission
    original=ROOT/'reports/compact-training.json'
    initial=json.loads(original.read_text())
    eligible,_,_=detector_admission(initial['baseline_validation'],initial['validation'])
    if eligible:
        print('Source validation already passes; the extra trial is skipped',flush=True)
        return
    plan=json.loads((ROOT/'reports/replay-plan.json').read_text())
    if plan['epochs']!=3 or plan['lr0']!=.0005:
        raise ValueError('Unexpected refinement plan')
    archived=ROOT/'reports/colab-initial-training.json'
    if archived.exists():
        raise ValueError('Refinement was already started; preserve that run')
    shutil.copy2(original,archived)
    shutil.copy2(ROOT/'reports/compact-training-curve.csv',ROOT/'reports/colab-initial-training-curve.csv')
    os.environ['ZONEGUARD_TRAINING_ENV']='google_colab'
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_large.py'),'--replay','--epochs','3','--batch','16','--workers','2'],cwd=ROOT,check=True)
    result=json.loads((ROOT/'reports/replay-training.json').read_text())
    result['current_stage_epochs']=result['actual_epochs']
    result['actual_epochs']+=initial['actual_epochs']
    result['stages']=[{'name':'compact','epochs':initial['actual_epochs'],'restored_epochs':initial['restored_epochs']},
                      {'name':'source_replay','epochs':result['current_stage_epochs'],'plan':'reports/replay-plan.json'}]
    result['limitations']+=' Source sampling was refined using validation only; initial-trial test outputs did not enter that decision. Elapsed times cover the final stage only.'
    original.write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (ROOT/'reports/colab-initial-training-curve.csv').open() as file:
        reader=csv.DictReader(file);columns=reader.fieldnames;rows=list(reader)
    with (ROOT/'reports/replay-training-curve.csv').open() as file:
        extra=list(csv.DictReader(file))
    for row in extra:
        row['epoch']=str(int(float(row['epoch']))+initial['actual_epochs'])
    with (ROOT/'reports/compact-training-curve.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows+extra)
    live=ROOT/'reports/training-live.json';state=json.loads(live.read_text())
    state.update(epoch=result['actual_epochs'],epochs=result['actual_epochs'],report='reports/compact-training.json')
    live.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('Completed validation-driven refinement',flush=True)


if __name__=='__main__':
    main()
