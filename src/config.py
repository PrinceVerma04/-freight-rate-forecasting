"""
Central configuration for the freight-rate project.

Every file path, random seed, constant and date cutoff lives here, so the rest
of the code never hard-codes a "magic number". Change a value here and the
whole pipeline picks it up.
"""

from pathlib import Path

import pandas as pd

# =============================================================================
# Paths
# =============================================================================
# The project root is the folder that contains this "src" folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The four provided input files live in data/, unchanged. NOTE: the provided
# readme calls them "data/train_test.csv" etc., but the real files use hyphens.
# We always use the real names.
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_INPUTS_PATH = DATA_DIR / "december-chart-inputs.csv"
SCORER_PATH = PROJECT_ROOT / "score.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SCORER_RESULTS_DIR = PROJECT_ROOT / "scorer_results"

# Deliverables. The brief asks for a file called exactly
# "validation_predictions.csv", so it is written to the project root (where
# the readme's scorer command looks for it). There is only one copy, so a
# reviewer never has to wonder which file is the submission.
VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"
# The original December input file is left untouched; the filled copy lives here.
DECEMBER_PREDICTIONS_PATH = OUTPUT_DIR / "december_chart_predictions.csv"

MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
FOLD_DETAILS_PATH = OUTPUT_DIR / "cv_fold_details.csv"
GROUP_IMPORTANCE_PATH = OUTPUT_DIR / "feature_group_importance.csv"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.joblib"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 42
# A fixed thread count keeps LightGBM / CatBoost results identical between runs.
N_THREADS = 4

# =============================================================================
# Data constants (each one was verified in the data, see reports/report.md)
# =============================================================================
DATA_START_DATE = pd.Timestamp("2025-01-01")  # first day in train-test.csv

WEIGHT_CAP_LBS = 47_500    # 1,204 loads sit at exactly this value -> a cap
WEIGHT_FLOOR_LBS = 5_000   # 32 loads sit at exactly this value -> a floor
MIN_DISTANCE_MILES = 70    # distances are floored at 70 miles

# A reported distance more than 3x (or less than 1/3) of the expected distance
# for that lane would be treated as corrupted and replaced.
DISTANCE_RATIO_LIMIT = 3.0

# Corrupted posted_rate detection: a label is flagged when
#   actual / expected  <  CORRUPT_RATIO_LOW   or   >  CORRUPT_RATIO_HIGH.
# In the data every clean load has a ratio between ~0.84 and ~1.21, while the
# corrupted ones sit below 0.5 or above 2.0. Both thresholds are in the empty
# gap between those clusters (see reports/figures/target_ratio_histogram.png).
CORRUPT_RATIO_LOW = 0.6
CORRUPT_RATIO_HIGH = 1.8

# Distance bands for the simple "reference price" model (median $/mile per band).
DISTANCE_BAND_EDGES = [70, 150, 250, 400, 600, 900, 1300, 1800, 2400, 3500]

# =============================================================================
# Feature engineering settings
# =============================================================================
N_REGIONS = 8                 # KMeans clusters of city coordinates
N_NEAREST_CITIES = 5          # neighbours used to estimate an unseen city's price level
CITY_LEVEL_SMOOTHING = 20     # pseudo-loads that pull a city's level towards 0
LANE_LEVEL_SMOOTHING = 20     # pseudo-loads that pull a lane's level towards its cities
HOLIDAY_WINDOW_DAYS = 3       # "holiday week" = within +/- 3 days of a holiday
HOLIDAY_DISTANCE_CAP_DAYS = 7 # days_to_nearest_holiday is capped at one week

# The holidays that matter for US trucking (shippers and drivers take time off).
# Training (Jan-Oct) contains New Year, Memorial Day, July 4 and Labor Day;
# the prediction period contains Thanksgiving, Christmas and New Year 2026.
US_FREIGHT_HOLIDAYS = pd.to_datetime([
    "2025-01-01",  # New Year's Day
    "2025-05-26",  # Memorial Day
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    "2026-01-01",  # New Year's Day
])

# =============================================================================
# Validation design (Phase 3)
# =============================================================================
# The real task is a forecast: train on Jan-Oct, predict Nov-Dec.
# So every split trains on the past and tests on a later period.
MAIN_HOLDOUT = {"name": "Holdout Sep-Oct", "train_end": "2025-08-31",
                "test_start": "2025-09-01", "test_end": "2025-10-31"}

# Expanding-window ("rolling origin") folds: always train on everything
# before the test month, then test on that one month.
ROLLING_FOLDS = [
    {"name": "Jul", "train_end": "2025-06-30", "test_start": "2025-07-01", "test_end": "2025-07-31"},
    {"name": "Aug", "train_end": "2025-07-31", "test_start": "2025-08-01", "test_end": "2025-08-31"},
    {"name": "Sep", "train_end": "2025-08-31", "test_start": "2025-09-01", "test_end": "2025-09-30"},
    {"name": "Oct", "train_end": "2025-09-30", "test_start": "2025-10-01", "test_end": "2025-10-31"},
]

RANDOM_SPLIT_TEST_FRACTION = 0.2  # only used to show why a random split is misleading
N_SIMULATED_UNSEEN_CITIES = 8     # validation.csv has 8 cities never seen in training

# Boosting models are trained with MAX_BOOSTING_ROUNDS trees once per fold, and
# the test fold is scored after every ROUND_CHECKPOINT. The checkpoint with the
# best MEAN error over the folds becomes the model's number of trees (one value
# for all folds, like LightGBM's own lgb.cv), and the final model reuses it,
# scaled up for its larger training set.
MAX_BOOSTING_ROUNDS = 3000
ROUND_CHECKPOINTS = list(range(250, MAX_BOOSTING_ROUNDS + 1, 250))

# =============================================================================
# Model settings (Phase 4)
# =============================================================================
LIGHTGBM_PARAMS = {
    "objective": "regression",   # L2 loss on the transformed target
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 0.0,
    "seed": RANDOM_SEED,
    "deterministic": True,       # same data + same params -> same model
    "force_col_wise": True,
    "num_threads": N_THREADS,
    "verbose": -1,
}

# Huber loss switches from squared to absolute error beyond this residual.
# On the log scale 0.1 means "about 10% off", so clean rows (noise of ~2%) get
# normal L2 treatment while corrupted rows (50%+ off) barely move the model.
HUBER_DELTA = 0.1

# CatBoost is ~10x slower than LightGBM on this data, so it gets a bounded
# budget: fewer, bigger steps and no automatic combinations of the text
# categories (those combinations made it a further ~1.5x slower).
CATBOOST_MAX_ROUNDS = 1500
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.15,
    "depth": 6,
    "max_ctr_complexity": 1,
    "boosting_type": "Plain",
    "random_seed": RANDOM_SEED,
    "thread_count": N_THREADS,
    "allow_writing_files": False,  # stops CatBoost creating a catboost_info/ folder
    "verbose": 0,
}

RIDGE_ALPHA = 1.0

# Light hyper-parameter tuning: a small grid around the defaults
# (num_leaves=31, min_child_samples=20, max_bin=255), scored on the rolling folds.
LIGHTGBM_TUNING_GRID = [
    {"num_leaves": 15, "min_child_samples": 20},
    {"num_leaves": 63, "min_child_samples": 50},
    {"num_leaves": 127, "min_child_samples": 50},
    {"max_bin": 1023},  # finer split points for distance and market_index
]

# Two results whose mean MAE differs by less than this fraction are treated as
# a tie, and the tie is broken with MAPE (the rule from the brief).
TIE_TOLERANCE = 0.002

# `python run_pipeline.py --fast` skips the feature ablation, target options,
# tuning and CatBoost, and uses these choices instead. The feature groups below
# come from the earlier exploratory experiments (quote_signal, market
# interactions, day_of_week and holidays all raised the time-based CV error).
# After a FULL run, copy its choices here so --fast reproduces it.
FAST_MODE_FEATURE_GROUPS = ["distance", "geography", "city_level", "equipment", "weight",
                            "market", "quarter_end", "time_trend"]
# e.g. {"base": "G_ridge_then_lightgbm", "target_kind": "log", "params": {"num_leaves": 63}}
FAST_MODE_TUNING = None
