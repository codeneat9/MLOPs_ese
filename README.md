# SmartWaste

SmartWaste is an MLOps application for classifying waste images into five
canonical classes:

`plastic`, `paper`, `glass`, `metal`, and `organic`.

The project is being implemented in phases. Dataset preparation and the first
EfficientNet-B0 candidate training/evaluation run are complete. API serving,
webcam classification, persistence, monitoring, model approval, cloud
integration, and deployment remain incremental work; unfinished features are
not presented as operational.

## Requirements

- Python 3.11
- Git
- Optional for later phases: Docker and Docker Compose

The original dataset is outside the repository. Processed data and generated
model/evaluation artifacts are local and ignored by Git. Do not silently map
unrelated labels such as a generic `trash` class to `organic`.

## Local setup

```bash
cd /Users/yashwithbehara/Desktop/SmartWaste
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dvc]"
cp .env.example .env
pytest
ruff check .
```

The `.env` file is local-only and is ignored by Git. Adjust paths or limits
there without committing secrets. The defaults use SQLite and local MLflow
file storage; later phases will add the database schema, training pipeline,
and service processes.

## Dataset layout

```text
data/raw/
├── plastic/
├── paper/
├── glass/
├── metal/
└── organic/
```

Dataset preparation validates readability, class membership, duplicates, class
coverage, and leakage-safe reproducible splits before training. No dataset
download is performed automatically.

With a dataset in the layout above, run:

```bash
python -m smartwaste.data
```

This writes `data/processed/validation_report.json`, a deterministic
`data/processed/manifest.csv`, and `train/`, `validation/`, and `test/`
directories. Duplicate files and invalid images stop preparation rather than
allowing possible train/test leakage.

After initializing DVC, the same operation can be reproduced with:

```bash
dvc init
dvc repro validate_and_prepare
```

For the supplied RealWaste dataset, use the explicit documented mapping
below without copying the original dataset into this repository:

```bash
python -m smartwaste.data \
  --source-dir "/Users/yashwithbehara/Downloads/realwaste-main/RealWaste" \
  --output-dir data/processed
```

The mapping is:

```text
Cardboard      -> paper
Food Organics  -> organic
Glass          -> glass
Metal          -> metal
Paper          -> paper
Plastic        -> plastic
Vegetation     -> organic
```

`Miscellaneous Trash` and `Textile Trash` are excluded because they do not
semantically belong to one of the five target classes. The RealWaste source
README identifies its license as CC BY-NC-SA 4.0; retain attribution and
observe the non-commercial/share-alike terms when distributing results.

For a local DVC remote, use a directory outside the repository, for example:

```bash
mkdir -p ../smartwaste-dvc-storage
dvc remote add -d local ../smartwaste-dvc-storage
dvc add data/raw
dvc push
```

## Phase roadmap

1. **Scaffolding:** configuration, dependency management, hygiene, and smoke
   tests.
2. **Data pipeline:** integrity checks, preparation, stratified splits, and
   DVC stages.
3. **Training/evaluation (complete locally):** EfficientNet-B0 transfer
   learning, checkpoints, metrics, and plots. MLflow tracking remains pending.
4. **Serving and persistence:** FastAPI inference, SQLAlchemy prediction logs,
   reviewed feedback, and model approval metadata.
5. **Monitoring and operations:** Streamlit/Plotly dashboard, webcam page,
   local drift reports, controlled retraining, Docker Compose, and CI/CD.

### Monitoring and controlled retraining

Create a reference prediction file with columns `predicted_class,confidence`,
then generate a local drift report from recent SQLite predictions:

```bash
python -m smartwaste.monitoring \
  --reference data/monitoring/reference.csv \
  --output artifacts/monitoring/drift.json
```

The report uses population stability index (PSI) for class-distribution drift
and mean-confidence change. PSI `0.20` and a confidence drop of `0.05` are the
default warning thresholds. The API exposes the report at
`GET /monitoring/summary`, and the dashboard displays warnings and distributions.

Reviewed feedback is eligible for retraining only after an explicit review
accepts it:

```bash
python -m smartwaste.retraining \
  --checkpoint models/candidate/best.pt \
  --metrics artifacts/evaluation/metrics.json \
  --baseline-metrics artifacts/evaluation_final_merged/metrics.json
```

This exports accepted feedback to `data/feedback/accepted.jsonl`, requires the
candidate to meet the default minimum of one accepted feedback record and
minimum accuracy/F1 thresholds, rejects regressions greater than two percentage
points against the approved baseline, and only then promotes the checkpoint
into `models/approved/`. Use `--minimum-feedback` to change the retraining
trigger.

### Container build and deployment workflow

Run the local deployment with:

```bash
docker compose up --build
```

The API is available on port 8000 and the dashboard on port 8501. The
`Container deployment` GitHub Actions workflow runs only after a successful CI
workflow and packages the tested Docker image as an Actions artifact. It does
not require or configure a cloud registry.

## Known compatibility risks

- PyTorch wheels are platform-specific; installation may require the CPU or
  accelerator wheel index appropriate for the developer machine.
- Evidently APIs vary across releases, so its integration will be isolated and
  validated against the installed version rather than assumed.
- MLflow registry behavior depends on the configured backend. Candidate
  promotion will remain explicit and validation-gated.
- Training and monitoring require a real labelled dataset; this repository
  uses the mapped RealWaste dataset described above.

## Training and evaluation

After preparing the dataset, train a candidate model using the parameters in
`params.yaml`:

```bash
python -m smartwaste.train
python -m smartwaste.evaluate
```

Training saves the best validation checkpoint and history under
`models/candidate/`. Evaluation uses only the held-out test split and writes
metrics, a confusion-matrix CSV, and a PNG plot under
`artifacts/evaluation/`. If `--tracking-uri` is supplied to training, the
parameters, epoch metrics, checkpoint, and history are logged to MLflow.
Pretrained EfficientNet weights may require a one-time network download; use
`pretrained: false` in `params.yaml` when testing offline.

The current measured test results and implementation status are documented in
[PROJECT_SPEC.md](./PROJECT_SPEC.md). The candidate checkpoint is not an
approved production model.

The current approved model was trained using the deduplicated
`final_merged_dataset` source. It contains 18,233 unique valid images after
mapping `cardboard` to `paper` and excluding `battery`, `textile`, and generic
`trash`. Its held-out test accuracy is 97.33% and macro F1 is 97.22%.
These measurements do not establish perfect performance on arbitrary
webcam or stock-product images.

## Model approval and rollback

The evaluated candidate is promoted explicitly into `models/approved/`; the
candidate checkpoint remains unchanged. To approve a future candidate only
after evaluation:

```bash
python -m smartwaste.governance approve \
  --checkpoint models/candidate/best.pt \
  --metrics artifacts/evaluation/metrics.json
```

The command enforces minimum accuracy and macro-F1 thresholds and preserves an
append-only promotion history. To restore the previous approved model:

```bash
python -m smartwaste.governance rollback
```

The API loads the approved manifest when present and falls back to the
candidate checkpoint only when no approved manifest exists. A newly started
API process is required after promoting or rolling back a model.

## FastAPI inference service

The existing candidate checkpoint can be served locally:

```bash
uvicorn smartwaste.api:app --host 0.0.0.0 --port 8000
```

Available verified endpoints:

```text
GET  /health
GET  /classes
POST /predict          multipart field: file
GET  /metrics/summary
GET  /predictions      recent prediction metadata
POST /feedback         submit a corrected label
POST /feedback/{id}/review
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/image.jpg"
```

`/predict` returns a request ID, predicted class, probabilities for all five
classes, checkpoint-derived model version, and inference latency. Files are
validated as readable images and are limited by
`SMARTWASTE_MAX_UPLOAD_BYTES`; image contents are not persisted.

Prediction metadata is stored in SQLite. Corrected labels start as unreviewed
and are not eligible for retraining until an explicit review request accepts
them.

## Streamlit dashboard

Start the API first, then open a second terminal:

```bash
cd /Users/yashwithbehara/Desktop/SmartWaste
source .venv/bin/activate
streamlit run src/smartwaste/dashboard.py
```

The dashboard provides image upload classification, confidence charts,
prediction history, measured evaluation metrics, corrected-label submission,
and a Live Camera page. The Live Camera page uses `streamlit-webrtc` for
continuous browser webcam frames, runs throttled local inference, overlays the
prediction on the video, and does not log every frame to SQLite. Set
`SMARTWASTE_API_URL` when the API is hosted at a different address.

To use live inference:

1. Keep the FastAPI terminal running on port 8000.
2. Start Streamlit with `streamlit run src/smartwaste/dashboard.py`.
3. If using an iPhone, place it near the Mac, keep it unlocked, sign in to
   both devices with the same Apple ID using two-factor authentication, and
   enable **Continuity Camera** on the iPhone under **Settings > General >
   AirPlay & Continuity > Continuity Camera**.
4. Open the Streamlit URL and select **Live Camera**.
5. Click **SELECT DEVICE**, choose **iPhone Camera** (or the desired camera),
   allow camera access, and click **START**.
6. Hold one waste item in view.

The callback throttles inference to avoid overlapping model calls. Video frames
are processed in memory and are not persisted. The Live Camera page intentionally
has no photo-capture fallback: it is a continuous video experience. If the
component is black, verify that camera permission is granted, confirm the camera
is selected in **SELECT DEVICE**, and try Chrome or Edge if Safari does not
render the WebRTC stream.

### Native laptop-camera mode

If browser webcam streaming is unavailable, use the native OpenCV camera
window. It connects directly to the laptop camera and draws the prediction on
each displayed frame:

```bash
./.venv/bin/pip install -e .
./.venv/bin/smartwaste-live-camera
```

Hold one waste item in front of the camera. Press `q` or `Esc` to close the
window. If the default camera is unavailable, try another macOS camera index:

```bash
./.venv/bin/smartwaste-live-camera --camera-index 1
```
