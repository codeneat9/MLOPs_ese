# SmartWaste Project Specification and Status

## Current architecture

```text
RealWaste source
    -> explicit class mapping and validation
    -> reproducible train/validation/test splits
    -> EfficientNet-B0 training
    -> candidate checkpoint and evaluation artifacts
    -> [next] approved model registry
    -> [next] FastAPI inference and Streamlit dashboard
    -> [next] prediction history, feedback, monitoring, and retraining
```

The current implementation is a local Python package under `src/smartwaste`.
It uses Pillow, scikit-learn, PyTorch/torchvision, DVC stage definitions,
and file-based artifacts. No cloud service, API, dashboard, database, or model
registry integration is currently verified.

## Dataset and provenance

The prepared dataset is derived from RealWaste at
`/Users/yashwithbehara/Downloads/realwaste-main/RealWaste`. The source README
states CC BY-NC-SA 4.0 licensing; attribution, non-commercial use, and
share-alike requirements must be preserved.

Explicit mapping:

```text
Cardboard      -> paper
Food Organics  -> organic
Glass          -> glass
Metal          -> metal
Paper          -> paper
Plastic        -> plastic
Vegetation     -> organic
```

`Miscellaneous Trash` and `Textile Trash` are excluded. The prepared dataset
contains 3,939 valid images, with no unreadable files or duplicate hash groups.

## Requirement-to-evidence checklist

| Area | Status | Evidence or blocker |
|---|---|---|
| Five-class dataset validation | VERIFIED | `src/smartwaste/data.py`; `data/processed/validation_report.json` |
| Reproducible stratified splits | VERIFIED | `data/processed/manifest.csv`; seed 42 |
| EfficientNet-B0 transfer learning | VERIFIED | `src/smartwaste/training.py`; merged-dataset approved checkpoint |
| Actual held-out evaluation | VERIFIED | `artifacts/evaluation_final_merged/metrics.json` |
| DVC data/training/evaluation pipeline | IN PROGRESS | `dvc.yaml` exists; DVC reproduction and remote are not verified |
| MLflow experiment tracking | NOT STARTED | Optional hook exists; no MLflow run evidence found |
| Approved model registry and rollback | VERIFIED | `src/smartwaste/governance.py`; two approved versions and promotion history |
| FastAPI inference API | VERIFIED | `src/smartwaste/api.py`; mocked tests and real checkpoint smoke test |
| Image upload interface | IMPLEMENTED | `src/smartwaste/dashboard.py`; API-backed upload view |
| Live webcam page | VERIFIED | `streamlit-webrtc` continuous frames with throttled local inference and prediction overlay |
| Prediction database/history | VERIFIED | SQLite/SQLAlchemy records and `GET /predictions` |
| Reviewed feedback workflow | VERIFIED | `POST /feedback` and explicit review endpoint |
| Monitoring and drift reports | NOT STARTED | Dashboard metrics are implemented; Evidently drift integration is not implemented |
| Controlled retraining gates | NOT STARTED | Candidate comparison and approval criteria are not implemented |
| AWS S3 artifact integration | NOT STARTED | Requires AWS credentials/bucket and live verification |
| Cloud API deployment | BLOCKED | No hosting account, service configuration, or deployment evidence |
| Docker Compose deployment | NOT STARTED | No Dockerfile or Compose file exists |
| GitHub Actions CI | IMPLEMENTED | `.github/workflows/ci.yml`; successful remote CI run is not verified |
| Architecture and setup documentation | IN PROGRESS | `README.md` and this specification; diagrams and demo evidence remain |
| Screenshots and presentation evidence | BLOCKED | Must be captured from the real completed application |

## Verified experimental results

The original candidate was evaluated on 591 held-out test images:

- Accuracy: `0.962775`
- Macro precision: `0.959574`
- Macro recall: `0.961205`
- Macro F1: `0.960178`

Per-class metrics and the confusion matrix are in
`artifacts/evaluation/metrics.json`. These are dataset-specific measurements,
not a guarantee of real-world accuracy. The merged-dataset candidate was evaluated on 2,735 held-out test images:

- Accuracy: `0.973309`
- Macro precision: `0.970374`
- Macro recall: `0.974299`
- Macro F1: `0.972246`

The merged candidate was promoted because it improved both accuracy and macro
F1 against the previous approved baseline. The previous approved checkpoint
remains in promotion history and can be restored.

The clean stock-style bottle screenshot still predicts as paper. This is an
unresolved domain-shift example because the screenshot is not representative
of the labeled training distribution and includes browser-context pixels.
No perfect-accuracy claim is made.

## Incremental implementation plan

1. Add model approval metadata and feedback boundaries around the tested
   FastAPI API for health, classes, prediction, and metrics summary.
2. Add dashboard views for SQLite/SQLAlchemy prediction history and reviewed feedback.
3. Add Streamlit upload classification and prediction-history views.
4. Add the Live Camera page using a supported webcam component with continuous
   frames, prediction overlays, and explicit permission/error handling.
5. Add MLflow run logging/registry workflow and connect controlled candidate
   approval with the dashboard.
6. Add monitoring, drift warnings, controlled retraining, Docker, cloud
   artifact configuration, and deployment documentation.

Each step must be tested before the next major feature begins. No cloud
integration or deployment will be described as working until tested against
the actual service.
