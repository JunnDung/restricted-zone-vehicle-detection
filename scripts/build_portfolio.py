"""Build portfolio evidence from measured results, never from proposed targets."""
import csv
import json
from pathlib import Path

from data_utils import sha256

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('mio','visdrone','coco')


def case_study(report):
    before=sum(report['baseline_test'][s]['mAP50_95'] for s in SOURCES)/3
    after=sum(report['test'][s]['mAP50_95'] for s in SOURCES)/3
    performance=(f'improved equal-source held-out detector mAP50–95 from {before:.1%} to {after:.1%} ({(after-before)*100:+.1f} percentage points)'
                 if after>before else f'benchmarked equal-source held-out detector mAP50–95 at {after:.1%} against a {before:.1%} baseline')
    gate='passed; candidate available in the dashboard' if report['accepted'] else 'failed; the previous serving model is retained'
    lines=['# ZoneGuard — engineering case study','',
           'Vehicle intrusion detection for fixed cameras: YOLO11n → ByteTrack → polygon/dwell rules → evidence and persistent alerts.','',
           '## Measured detector results','',
           '| Held-out subset | Baseline mAP50–95 | Candidate mAP50–95 | Change |',
           '|---|---:|---:|---:|']
    for source in SOURCES:
        old=report['baseline_test'][source]['mAP50_95'];new=report['test'][source]['mAP50_95']
        lines.append(f'| {source} | {old:.2%} | {new:.2%} | {(new-old)*100:+.2f} pp |')
    lines.extend([f'| Equal-source mean | {before:.2%} | {after:.2%} | {(after-before)*100:+.2f} pp |','',
                  f"Deployment gate: **{gate}**. Checkpoint selection and promotion were decided using validation and the calibration video before inspecting test scores.",'',
                  f"Model: {report['parameters']:,} parameters; {report['weights_bytes']/1024**2:.2f} MiB inference checkpoint. Evaluation device: {report['device']}; image size {report['settings']['imgsz']}; validation/test batch {report['settings']['batch']}.",'',
                  '| Parking-video regression | Baseline | Candidate |','|---|---:|---:|'])
    video=report['admission_video']
    for label,field in (('Class-aware event F1','class_aware'),('Alert-only event F1','alert_only')):
        lines.append(f"| {label} | {video['baseline'][field]['f1']:.2f} | {video['candidate'][field]['f1']:.2f} |")
    lines.append(f"| Pipeline FPS, including video encoding | {video['baseline']['processing_fps']:.2f} | {video['candidate']['processing_fps']:.2f} |")
    lines.extend(['','The video contains only two reviewed events and was used for calibration. Its F1 is a regression result, not field accuracy. FPS is one full-video run per model on the stated device, not a cross-machine comparison.','',
                  '## What the implementation demonstrates','',
                  '- Five-class data conversion, ignored-region masking, deterministic sampling, exact-duplicate removal and SHA-256 provenance across 15,000 images (12,000/1,500/1,500 train/val/test).',
                  '- Cross-platform recovery of the original dataset and optimizer checkpoint; Colab T4 training resumes after five completed laptop epochs.',
                  '- Equal-source validation checkpoint selection with per-domain regression limits; both old and candidate models evaluated on held-out subsets.',
                  '- One alert per track/zone/entry, boundary hysteresis, class stabilization, evidence images, H.264 video, SQLite history, CSV/JSON exports and API/rule tests in CI.','',
                  '## CV wording (English)','',
                  '- Built an end-to-end vehicle intrusion monitoring system with YOLO11n, ByteTrack, editable polygon zones, FastAPI and SQLite, including evidence capture and video/CSV exports.',
                  f'- Curated a checksum-verified 15,000-image traffic dataset; {performance} on the selected MIO/VisDrone/COCO subsets.',
                  '- Implemented resumable Colab training, per-domain model admission gates and automated API/rule regression checks with GitHub Actions.','',
                  '## Limits and reproducibility','',
                  report['limitations'],'',
                  'This is a curated-subset experiment, not an official COCO/MIO/VisDrone benchmark. MIO camera independence is unverified; exact hashes do not remove near-duplicate scenes. COCO was used during upstream model development. No independent deployment-camera alert benchmark has been collected.','',
                  'Training elapsed time in the JSON report covers the resumed session, not the five earlier laptop epochs. Batch and hardware changed on resume, so training is not bitwise reproducible across machines.','',
                  'Evidence: [machine-readable report](../reports/compact-training.json), [curve](../reports/compact-training-curve.csv), [model card](../models/MODEL_CARD.md), [Colab procedure](COLAB.md).','',
                  '![Measured results](../reports/portfolio-results.png)',''])
    return '\n'.join(lines)


def main():
    report=json.loads((ROOT/'reports/compact-training.json').read_text())
    weights=ROOT/('models/vehicle-large.pt' if report['accepted'] else 'runs/zoneguard_compact/weights/deployment.pt')
    if sha256(weights)!=report['weights_sha256']:
        raise ValueError('Portfolio weights do not match the evaluated candidate')
    (ROOT/'docs/CASE_STUDY.md').write_text(case_study(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (ROOT/'reports/compact-training-curve.csv').open() as file:
        rows=[{key.strip():value for key,value in row.items()} for row in csv.DictReader(file)]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    axes[0].plot([int(float(r['epoch'])) for r in rows],[float(r['metrics/mAP50-95(B)'])*100 for r in rows],marker='o',color='#087f8c')
    axes[0].axvline(5,color='gray',linestyle='--',label='Resume on Colab after epoch 5')
    axes[0].set(xlabel='Epoch',ylabel='Validation mAP50-95 (%)',title='Mixed validation during training')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    x=list(range(3));width=.35
    axes[1].bar([v-width/2 for v in x],[report['baseline_test'][s]['mAP50_95']*100 for s in SOURCES],width,label='Baseline',color='#9aa5b1')
    axes[1].bar([v+width/2 for v in x],[report['test'][s]['mAP50_95']*100 for s in SOURCES],width,label='Candidate',color='#087f8c')
    axes[1].set(xticks=x,xticklabels=SOURCES,ylabel='mAP50-95 (%)',title='Held-out curated subsets')
    axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('ZoneGuard — measured detector evidence; not field alert accuracy')
    fig.savefig(ROOT/'reports/portfolio-results.png',dpi=180);plt.close(fig)
    print('Generated docs/CASE_STUDY.md and reports/portfolio-results.png from verified weights')


if __name__=='__main__':
    main()
