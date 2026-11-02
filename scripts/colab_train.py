"""Restore the verified Windows run on Colab and continue with a T4-sized batch."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from data_utils import sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Select a free T4 GPU runtime before starting this script')
    pause=json.loads((ROOT/'models/checkpoints/compact-last.json').read_text())
    source=ROOT/pause['checkpoints']['last']['backup']
    if sha256(source)!=pause['checkpoints']['last']['sha256']:
        raise ValueError('Archived checkpoint checksum differs')
    subprocess.run([sys.executable,str(ROOT/'scripts/restore_compact.py')],cwd=ROOT,check=True)
    run=ROOT/'runs/zoneguard_compact'
    last=run/'weights/last.pt'
    if not last.exists():
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        if checkpoint['epoch']+1!=pause['completed_epochs'] or checkpoint.get('optimizer') is None:
            raise ValueError('Checkpoint is not resumable at the recorded epoch')
        args=checkpoint['train_args']
        args.update(data=str(ROOT/'datasets/compact/data.yaml'),project=str(ROOT/'runs'),
                    name='zoneguard_compact',save_dir=str(run),model=str(ROOT/'models/vehicle-yolo11n.pt'))
        last.parent.mkdir(parents=True,exist_ok=True)
        torch.save(checkpoint,last)
        torch.save(checkpoint,last.with_name('best.pt'))
        history=checkpoint.get('train_results',{})
        if history:
            with (run/'results.csv').open('w',newline='') as file:
                writer=csv.writer(file);writer.writerow(history)
                writer.writerows(zip(*history.values()))
    else:
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['epoch']+1<pause['completed_epochs'] or checkpoint.get('optimizer') is None:
            raise ValueError('Recovery checkpoint is incomplete or training already finished')
        checkpoint['train_args'].update(data=str(ROOT/'datasets/compact/data.yaml'),project=str(ROOT/'runs'),name='zoneguard_compact',save_dir=str(run))
        torch.save(checkpoint,last)
    os.environ['ZONEGUARD_TRAINING_ENV']='google_colab'
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_large.py'),'--resume','--batch','16','--workers','2'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
