"""Rebuild the deleted compact dataset and require the archived content hashes."""
import json
import subprocess
import sys
from pathlib import Path

from data_utils import sha256, verify_files

ROOT=Path(__file__).resolve().parents[1]


def archived_writer(path,text,encoding=None,errors=None,newline=None):
    # The archived Windows manifests include CRLF in text files and their hashes.
    with path.open('w',encoding=encoding,errors=errors,newline='\r\n') as file:
        return file.write(text)


def main():
    expected=json.loads((ROOT/'reports/compact-run-settings.json').read_text())['dataset_manifest_sha256']
    target=ROOT/'datasets/compact'
    if not (target/'manifest.json').exists():
        stages=[('scripts/prepare_visdrone.py',ROOT/'data/vehicles/manifest.json'),
                ('scripts/prepare_large.py',ROOT/'datasets/large/manifest.json'),
                ('scripts/prepare_compact.py',target/'manifest.json')]
        for script,manifest in stages:
            if manifest.exists():
                continue
            print('Rebuilding '+script,flush=True)
            code="import runpy; from pathlib import Path; from restore_compact import archived_writer; Path.write_text=archived_writer; runpy.run_path("+repr(str(ROOT/script))+",run_name='__main__')"
            subprocess.run([sys.executable,'-c',code],cwd=ROOT/'scripts',check=True)
    if sha256(target/'manifest.json')!=expected:
        raise ValueError('Restored dataset differs from the archived run; refusing to resume. Keep the new data for a separately evaluated experiment.')
    manifest=json.loads((target/'manifest.json').read_text())
    verify_files(target,manifest['files'])
    print(json.dumps({'restored_images':len(manifest['files']),'manifest_sha256':expected,'resume_compatible':True}),flush=True)


if __name__=='__main__':
    main()
