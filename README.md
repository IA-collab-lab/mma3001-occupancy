# Predicting room occupancy (Building 69, G.57)

MMA3001 individual project. Predicts whether a monitored room will be occupied 1 hour ahead (and optionally 24 hours ahead) from historical occupancy sensor events.

## Acceptance criterion

The gradient boosting model passes if its test F1 exceeds the stronger baseline's test F1 by at least 0.03. The margin was chosen after viewing validation results (persistence F1 0.719, gradient boosting F1 0.764) and before the test period was scored.

## Install
Python 3.10 or newer.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Data
The Monash Dataset 2 occupancy CSV is not included because of its size (1.6 GB). Put it in a local `data/` folder. `data/` is ignored by git. `occupancy.io.csv_to_parquet(csv_path, out_path)` cleans it in chunks and writes a compact parquet file. The cleaning counts are saved in `reports/data_audit.txt`.

## Run
Arguments for the experiment script:

```text
usage: experiment.py [-h] [--parquet PARQUET] [--horizon HORIZON] [--final]

options:
  -h, --help         show this help message and exit
  --parquet PARQUET
  --horizon HORIZON
  --final
```

Validation run: `python -m occupancy.experiment --horizon 4`. Test run (scores the final period once): add `--final`. Results are written to `reports/`.

## Tests
    pytest --html=reports/test_report.html --self-contained-html
The 13 tests cover cleaning, timezone handling, the gap rule, feature timing, the time split and the metrics. The last report is `reports/test_report.html`.

## Documentation
HTML documentation for every function is in `docs/` (open `docs/index.html`). It is generated with `pdoc -o docs occupancy`.

## Limitations
- Only occupancy history is used. The environmental data could not be linked to the occupancy rooms.
- Five rooms exist, but only four have validation and test data. One room is scored as a held-out room.
- Test F1 varies by room (0.543 to 0.827).
- The test period includes the university summer break.
- Hyperparameters, the 72-hour gap rule and the grid size were not tuned.

## AI use
Claude (Anthropic) was used heavily to write the code, tests and report drafts. See `ai_log.md` and the commit messages.

## Depth sweep
`python -m occupancy.depth_sweep` compares training and validation F1 at several tree depths (validation data only). Output: `reports/depth_sweep_h4.csv`.
