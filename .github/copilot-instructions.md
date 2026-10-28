# SmartWaste contribution instructions

- Preserve existing checkpoints, processed data, evaluation artifacts, and
  experiment records. Never restart or overwrite a training run without an
  explicit request.
- Use the canonical class order: `plastic`, `paper`, `glass`, `metal`,
  `organic`.
- Keep dataset mappings explicit and documented. Never silently map
  ambiguous source classes to `organic`.
- Do not commit datasets, model binaries, MLflow stores, secrets, or local
  databases.
- Treat a trained candidate as unapproved until evaluation and configured
  acceptance criteria are recorded.
- Report actual measured metrics separately from confidence scores, drift
  indicators, and simulated examples.
- Add focused pytest coverage for every new API, persistence, dashboard, or
  retraining behavior.
- Do not claim cloud, deployment, webcam, CI, or MLflow verification without
  evidence from the real service or run.
