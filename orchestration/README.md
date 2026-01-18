Orchestration (Phase 8) — Quick Start

This folder contains a minimal pipeline runner used as the Phase 8 scaffold.

Run the pipeline (dry-run):

```bash
python -m orchestration.pipeline --steps all --dry-run
```

Run a subset of steps:

```bash
python -m orchestration.pipeline --steps ingest,preprocess --dry-run
```

Return codes:
- `0` — success
- `2` — unknown step
- `3` — step exception

Next actions:
- Wire `ingest`, `preprocess`, `train`, `evaluate`, `deploy` to real modules.
- Add `tests/integration/test_orchestration.py` to validate CLI wiring.
- Integrate into CI pipeline.
