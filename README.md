# Freight Rate Prediction - Spotter ML Engineer Assessment

Predict the price of a truck load (`posted_rate`, USD) for 12,000 future loads (Nov-Dec 2025),
learning from 48,000 labelled loads (Jan-Oct 2025). One command runs the whole pipeline, from the
raw CSV files to the submission file and the scorer's December chart.

**Final model:** Ridge regression on log(price), followed by LightGBM trained on Ridge's errors.
**Accuracy:** an average error of **$31.86 per load (1.36%)** on time-based validation - 66% below a
simple baseline - and $32.73 (1.39%) when forecasting two months ahead.

---

## Start here: the deliverables

| Deliverable asked for in the brief | Where it is |
|---|---|
| Validation predictions (`load_id,predicted_rate`, 12,000 rows) | [`validation_predictions.csv`](validation_predictions.csv) |
| Fixed December chart produced by the provided `score.py` | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| Report: validation approach, data split, results, December chart | [`reports/report.pdf`](reports/report.pdf) (also readable on GitHub: [`reports/report.md`](reports/report.md)) |
| Solution code, dependencies, run instructions | this repository - see [How to run](#how-to-run) |
| 2-3 minute Loom walkthrough | *link to be added* |

---

## Where to find what

The table follows the topics of the brief, so each question points to the code that answers it
and to the evidence it produced.

| Question | Code | Evidence |
|---|---|---|
| What did the data show? | [`src/eda.py`](src/eda.py) | report section 2, [`reports/figures/`](reports/figures) |
| Which data-quality issues were found, and how were they fixed? | [`src/data_cleaning.py`](src/data_cleaning.py) - one function per issue | report section 3, [`target_ratio_histogram.png`](reports/figures/target_ratio_histogram.png) |
| Which features does the model use? | [`src/features.py`](src/features.py) - `FEATURE_GROUPS` at the top of the file | |
| How was the data split for training and validation? | [`src/validation.py`](src/validation.py), fold dates in [`src/config.py`](src/config.py) (`ROLLING_FOLDS`, `MAIN_HOLDOUT`) | [`outputs/cv_fold_details.csv`](outputs/cv_fold_details.csv) |
| Which models were compared, and why was the winner chosen? | [`src/models.py`](src/models.py) (models A-G), [`src/experiments.py`](src/experiments.py) (selection rule: `pick_best`) | [`outputs/model_comparison.csv`](outputs/model_comparison.csv), [`model_comparison.png`](reports/figures/model_comparison.png) |
| How were the final predictions made? | [`src/predict.py`](src/predict.py) | [`validation_predictions.csv`](validation_predictions.csv), [`outputs/run_summary.json`](outputs/run_summary.json) |
| How was the December chart made? | [`src/december.py`](src/december.py) | [`outputs/december_chart_predictions.csv`](outputs/december_chart_predictions.csv), [`december_predictions_diagnostic.png`](reports/figures/december_predictions_diagnostic.png) |
| Where are the settings (seed, thresholds, model parameters)? | [`src/config.py`](src/config.py) | |
| What is the order of the whole pipeline? | [`run_pipeline.py`](run_pipeline.py) - one function per phase | |

---

## Project structure

```
.
├── README.md                    # this file
├── run_pipeline.py              # ONE command that runs phases 0-7 (start reading the code here)
├── score.py                     # scorer provided by Spotter (unchanged)
├── requirements.txt             # scorer pins + modelling libraries
├── requirements-lock.txt        # exact versions used to produce the submitted files
├── validation_predictions.csv   # SUBMISSION FILE (load_id, predicted_rate)
│
├── data/                        # the four files provided by Spotter (unchanged)
│   ├── train-test.csv                    # 48,000 labelled loads, Jan-Oct 2025
│   ├── validation.csv                    # 12,000 loads to predict, Nov-Dec 2025
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv         # one fixed load on every day of December
│
├── src/                         # all solution code, in pipeline order
│   ├── config.py                # paths, seed, thresholds, fold dates, model settings
│   ├── eda.py                   # Phase 0 - exploration facts and figures
│   ├── data_cleaning.py         # Phase 1 - one function per data-quality issue
│   ├── features.py              # Phase 2 - FeatureBuilder (fitted on training rows only)
│   ├── validation.py            # Phase 3 - time-based splits, metrics, cross-validation
│   ├── models.py                # Phase 4 - models A-G behind one fit/predict interface
│   ├── experiments.py           # Phase 4 - feature screen, model comparison, ablations, tuning
│   ├── predict.py               # Phase 5 - final training + validation predictions
│   ├── december.py              # Phase 6 - December chart predictions
│   └── plotting.py              # one consistent chart style for every figure
│
├── outputs/                     # everything the pipeline produces (except the submission file)
│   ├── model_comparison.csv     # one row per experiment: mean CV error per fold + holdout
│   ├── feature_group_importance.csv  # how much the error rises when a feature group is shuffled
│   ├── cv_fold_details.csv      # every metric, for every experiment, on every fold
│   ├── run_summary.json         # key numbers of the run (used to write the report)
│   ├── december_chart_predictions.csv  # filled copy of data/december-chart-inputs.csv
│   ├── final_model.joblib       # the trained final model + fitted cleaning / feature steps
│   └── run_log.txt              # full console output of the run that produced these files
│
├── reports/
│   ├── report.pdf               # the written report (submitted version)
│   ├── report.docx              # the same report, editable
│   ├── report.md                # the same report, readable on GitHub
│   └── figures/                 # every chart used in the report
│
├── scorer_results/
│   └── candidate_december.png   # chart created by score.py
│
└── docs/                        # the original assessment readme and requirements
```

---

## How to run

**1. Set up** (tested with Python 3.13 on Windows; any Python 3.10+ should work)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` keeps the scorer's original pins (`matplotlib`, `numpy`, `pandas<3`) and adds
scikit-learn, LightGBM, CatBoost and joblib. Use `requirements-lock.txt` for the exact versions.

**2. Run everything**

```bash
python run_pipeline.py          # full run: every experiment (~45 min on a 4-core laptop)
python run_pipeline.py --fast   # ~10 min: skips the feature screen, ablations, tuning and CatBoost,
                                # and rebuilds the same final model from the choices stored in src/config.py
```

The pipeline prints a summary after every phase and ends by running the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions outputs/december_chart_predictions.csv
```

| Phase | Code | Output |
|---|---|---|
| 0. Explore the data | `src/eda.py` | facts in the log, figures in `reports/figures/` |
| 1. Clean the data | `src/data_cleaning.py` | cleaning log with counts |
| 2. Engineer features | `src/features.py` | 37 features in 12 groups |
| 3. Time-based validation design | `src/validation.py` | 4 rolling monthly folds + a Sep-Oct check |
| 4. Model comparison | `src/experiments.py`, `src/models.py` | `outputs/model_comparison.csv`, `outputs/cv_fold_details.csv` |
| 5. Retrain the winner, predict validation | `src/predict.py` | `validation_predictions.csv`, `outputs/final_model.joblib` |
| 6. Predict the December chart inputs | `src/december.py` | `outputs/december_chart_predictions.csv` |
| 7. Run the provided scorer | `run_pipeline.py` | `scorer_results/candidate_december.png` |

> **Note on file names.** The assessment readme refers to `data/train_test.csv` etc. The files that
> were actually supplied use hyphens (`train-test.csv`, `validation.csv`,
> `validation-predictions-template.csv`, `december-chart-inputs.csv`); they are kept, unchanged,
> in `data/`. `data/december-chart-inputs.csv` is never modified - the filled copy is written to
> `outputs/december_chart_predictions.csv`, which is the file passed to `score.py`.

---

## Approach in short

**1. Validation that matches the task.** The loads to predict (Nov-Dec) all lie in the future of the
labelled data (Jan-Oct), so every check trains on the past and tests on a later period:
* **4 rolling monthly folds** - train Jan-Jun -> test Jul, Jan-Jul -> Aug, Jan-Aug -> Sep,
  Jan-Sep -> Oct. Models are chosen by their mean error over these folds.
* **A two-month-ahead check** - train Jan-Aug -> test Sep-Oct, to see how the error grows when
  forecasting further ahead.
* A **random 80/20 split** is computed only for contrast: it looks about 8% better, because random
  test loads share days (and the daily price level) with training loads, which never happens when
  forecasting the future.
* Every split re-fits its cleaning and feature statistics on its own training rows, so nothing leaks
  from test to train. Errors are measured on clean labels (corrupted prices excluded).

**2. Data cleaning** - one function per issue, statistics learned from training rows only:
negative weights made positive (sign-flip errors), missing weights filled with the equipment-type
median, weights capped at 47,500 lb flagged, missing `market_index` filled with the same-day median,
and 677 corrupted prices (1.41%; 0.15-0.5x or 2-6x the expected price) removed from training.
Suspiciously long distances were checked and turned out to be genuine 70-mile minimum loads.

**3. Features** - distance, geography (coordinates, direction of travel, regions), equipment, weight,
`market_index`, a quarter-end ramp and a time trend. Feature groups that raised the time-based error
were dropped: `quote_signal`, market interactions, day of week, holidays, and the target-encoded city
price levels (city prices follow location, so the coordinates already carry them - this also covers
the 8 cities that appear only in validation).

**4. Models** - A: a simple reference-price baseline, B: Ridge regression, C: LightGBM,
D: LightGBM with Huber loss trained on the corrupted rows too, E: CatBoost (full run only),
F: averages of the best 2-3 models, G: LightGBM that learns the corrections to Ridge's prediction.
All work on log(price), so errors are relative. The winner has the lowest mean MAE over the rolling
folds; results within 0.2% count as a tie, which is broken by MAPE.

**5. Final model (G)** - Ridge regression predicts log(price) first, including the 2025 price trend;
LightGBM then learns only Ridge's errors, without the trend feature, so the trend can be carried into
Nov-Dec (trees alone can only repeat the last month's level - they missed October, the start of a
new quarter, by $45-48 per load). It is retrained on all 47,323 clean Jan-Oct loads with 2,330 trees
(the number chosen by cross-validation, scaled up for the larger training set).

---

## Results

From the full run (`python run_pipeline.py`). Mean error over the 4 rolling monthly folds, on clean
labels (lower is better):

| Model | CV MAE ($) | CV MAPE | Oct fold MAE ($) | Sep-Oct check MAE ($) |
|---|---:|---:|---:|---:|
| A - reference-price baseline | 94.56 | 4.12% | 87.23 | 119.54 |
| B - Ridge regression | 36.65 | 1.57% | 37.38 | 37.75 |
| C - LightGBM | 36.20 | 1.57% | 45.41 | 44.61 |
| D - LightGBM, Huber loss, corrupted rows kept | 38.20 | 1.68% | 46.05 | 46.60 |
| E - CatBoost | 36.40 | 1.57% | 47.73 | 44.25 |
| F - average of G + C | 32.83 | 1.41% | 39.55 | 37.02 |
| G - Ridge, then LightGBM on Ridge's errors | 32.17 | 1.37% | 35.25 | 32.78 |
| **G, without city target encoding, `max_bin`=1023 (final)** | **31.86** | **1.36%** | **35.44** | **32.73** |

* The final model's error is **66% lower than the baseline** and 12% lower than plain LightGBM.
* A random 80/20 split scores $30.28 - it looks better than the honest $32.73 only because of the
  leakage described above.
* For loads touching 8 cities removed from training, the error is $38.60 (1.57%), against $33.62
  (1.36%) for the same loads when those cities were seen.
* The December chart rises from $834 to $878: a weekly `market_index` cycle plus the quarter-end ramp.

Full numbers: [`outputs/model_comparison.csv`](outputs/model_comparison.csv),
[`outputs/run_summary.json`](outputs/run_summary.json) and the report.

---

## Reproducibility

* One fixed random seed (`RANDOM_SEED = 42` in `src/config.py`) is passed to every model, KMeans and split.
* LightGBM runs with `deterministic=True` and a fixed thread count; CatBoost with a fixed seed and thread count.
* Every statistic learned from data (median weights, lane distances, reference prices, regions,
  city price levels) is fitted on the training rows of each split only - see the "leakage" notes in
  `src/data_cleaning.py` and `src/features.py`.
