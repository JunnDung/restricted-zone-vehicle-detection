"""Install the training environment inside an ephemeral Colab runtime."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def status(phase,**details):
    report={'phase':phase,'updated_utc':datetime.now(timezone.utc).isoformat(),**details}
    path=ROOT/'reports/colab-bootstrap.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)
    print(json.dumps(report),flush=True)


def main():
    status('installing_cuda_dependencies')
    subprocess.run([sys.executable,'-m','pip','install','torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements.txt'),'pillow==12.3.0','pyyaml==6.0.3'],check=True)
    import torch,ultralytics,PIL
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available after installation')
    runtime={'python':sys.version.split()[0],'torch':torch.__version__,'ultralytics':ultralytics.__version__,'pillow':PIL.__version__,
             'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}
    (ROOT/'reports/colab-runtime.json').write_text(json.dumps(runtime,indent=2))
    status('restoring_dataset_and_resuming',runtime=runtime)
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/colab_train.py')],cwd=ROOT,check=True)
    status('completed')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('error',error=str(error));raise
