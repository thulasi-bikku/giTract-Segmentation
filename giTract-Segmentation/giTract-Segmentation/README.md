# giTract-Segmentation

Code for *Composite Difficulty Guided Curriculum Learning for Multi-Organ Gastrointestinal Tract Segmentation in
Abdominal MRI*. The repository contains the building blocks of the paper (composite slice difficulty, three-stage
Easy to Hard curriculum, Focal + Dice loss, AGTSNet) and a stricter evaluation protocol for follow-up studies.

## Status of the two protocols

* **Protocol used for the numbers in the paper**: slice-level random partition, strata that include the baseline
  Dice (S4), single run. It is documented in the manuscript and its limitations are stated there. The original
  scripts are in the archived release (Zenodo DOI in the manuscript).
* **Protocol in this folder (`scripts/`)**: patient-disjoint partition, baseline-independent test strata, train-only
  normalization, out-of-fold S4, control schedules, several seeds, patient-level cluster-bootstrap confidence
  intervals. It has **not** been used to produce any number in the manuscript.

`gitract/models.py` re-implements AGTSNet from the architecture description (residual blocks with SE, ASPP
bottleneck, attention-gated skips). If you have the original model file, replace this module; the pipeline only needs
`AGTSNet(in_ch=1, n_classes=3)` returning logits. `FocalDiceLoss` uses common defaults (alpha 0.25, gamma 2, eps 1e-6);
set the values of your released implementation if they differ.

## Install

```bash
pip install -r requirements.txt
pytest -q                       # CPU-only unit tests of splits, scoring, schedules and statistics
```

## Data

Download the UW-Madison GI Tract Image Segmentation data from Kaggle
(`happyharrycn/uw-madison-gi-tract-image-segmentation-dataset`) so that `train.csv` and `train/` sit in one folder.

## Run

```bash
export PYTHONPATH=.
python scripts/01_build_index.py --data-root /path/to/uw-madison --out data/index.csv

# patient split + independent strata (S1-S3 only, constants estimated on training patients)
python scripts/02_make_splits_and_strata.py --index data/index.csv --out data/index_split.csv

# optional: out-of-fold S4 for the training schedule, then rebuild strata with it
python scripts/03_train_oof_baseline.py --index data/index_split.csv --out data/oof_dice.csv
python scripts/02_make_splits_and_strata.py --index data/index.csv --oof-dice data/oof_dice.csv --out data/index_split.csv

# train every schedule over several seeds, then evaluate on the held-out test patients
for sched in baseline curriculum reversed random_stage matched_exposure oversample_hard; do
  for seed in 0 1 2 3 4; do
    python scripts/04_train.py --schedule $sched --seed $seed --out runs/${sched}_s${seed}
    python scripts/05_evaluate.py --run runs/${sched}_s${seed}
  done
done

# per-stratum means with patient-level CIs, and the paired difference curriculum - baseline
python scripts/06_analyze.py --runs "runs/curriculum_s*" --ref "runs/baseline_s*"
```

## Design notes

* `stratum_eval` (test reporting) uses S1-S3 only; no trained model enters its definition.
* `stratum_train` (training schedule) may add S4 computed out of fold.
* Thresholds are training-set quantiles that reproduce the paper prevalences (79.1 / 15.5 / 5.4 %).
* All test slices of the held-out patients are evaluated, so group results are on the natural distribution.
* Control schedules: `random_stage`, `reversed`, `matched_exposure` (single stage, presentation counts 75/50/25),
  `oversample_hard`. `04_train.py` writes `exposure.json` with the expected presentations per slice and the
  dataset-equivalent epochs (Eqs. 10 and 11 of the paper).
* The test partition is read only by `05_evaluate.py`; thresholds and checkpoints are fixed from training and
  validation data (the final checkpoint is used).
* Metrics: Dice, IoU and HD95 (pixels), averaged over the annotated organs of each slice.

## License

MIT.
