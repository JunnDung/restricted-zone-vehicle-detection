"""Install the training environment inside an ephemeral Colab runtime."""
import json
import subprocess
import sys
import venv
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
    environment=ROOT/'.colab-venv'
    # Colab may omit ensurepip; pip is inherited without changing the kernel packages.
    venv.EnvBuilder(with_pip=False,system_site_packages=True).create(environment)
    python=str(environment/'bin/python')
    subprocess.run([python,'-m','pip','install','torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    subprocess.run([python,'-m','pip','install','-r',str(ROOT/'requirements-colab.txt')],check=True)
    probe="import sys,json,torch,PIL,importlib.metadata as m; assert torch.cuda.is_available(), 'CUDA is not available'; print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'ultralytics':m.version('ultralytics'),'pillow':PIL.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory}))"
    runtime=json.loads(subprocess.check_output([python,'-c',probe],text=True))
    (ROOT/'reports/colab-runtime.json').write_text(json.dumps(runtime,indent=2))
    status('restoring_dataset_and_resuming',runtime=runtime)
    subprocess.run([python,'-u',str(ROOT/'scripts/colab_train.py')],cwd=ROOT,check=True)
    status('completed')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('error',error=str(error));raise
