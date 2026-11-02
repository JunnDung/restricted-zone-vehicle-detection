"""Package code, the trusted checkpoint and demo assets without Git credentials."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    weights={'models/vehicle-yolo11n.pt','models/checkpoints/compact-last.pt'}
    files=[name for name in files if not name.endswith('.png') and (not name.endswith('.pt') or name in weights)]
    recovery=['runs/zoneguard_compact/weights/'+name for name in ('last.pt','best.pt','balanced-best.pt')]
    recovery+=['runs/zoneguard_compact/results.csv','reports/colab-checkpoint-selection.json','reports/colab-baseline-validation.json']
    files=list(dict.fromkeys(files+[name for name in recovery if (ROOT/name).is_file()]))
    metadata={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
              'files_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    output=ROOT/'dist/zoneguard-colab.zip';output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name in files:
            archive.write(ROOT/name,name)
        archive.writestr('COLAB_PACKAGE.json',json.dumps(metadata,indent=2))
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
