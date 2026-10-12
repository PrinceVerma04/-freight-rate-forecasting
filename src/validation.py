"""
Phase 3 - validation strategy: how we split the data and how we score models.

The real task is a FORECAST: learn from Jan-Oct 2025, then predict Nov-Dec
2025. So every evaluation trains on the past and tests on a LATER period:

  * rolling folds  - train Jan-Jun -> test Jul, Jan-Jul -> Aug,
                     Jan-Aug -> Sep, Jan-Sep -> Oct   (used to pick the model)
  * main holdout   - train Jan-Aug -> test Sep-Oct    (a two-month-ahead check)

A random split is implemented too, but ONLY to show how misleadingly
optimistic it is: with random rows, the model sees loads from the very same
days it is tested on, which can never happen when forecasting the future.

Each split is cleaned and feature-engineered from its own training rows
(prepare_model_inputs), so no statistic ever leaks from test to train.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.features import ModelInputs, prepare_model_inputs
from src.models import build_model, select_training_rows


# =============================================================================
# Splits
# =============================================================================
@dataclass
class Fold:
    """One train/test split with its model-ready tables."""
    name: str
    inputs: ModelInputs


def split_by_dates(df, fold_definition):
    """
    Split rows by date.

    Input:  DataFrame + dict with train_end, test_start, test_end (strings).
    Output: (train_rows, test_rows) - training rows are strictly before the test period.
    """
    train_rows = df[df["date"] <= fold_definition["train_end"]]
    is_test = (df["date"] >= fold_definition["test_start"]) & (df["date"] <= fold_definition["test_end"])
    return train_rows, df[is_test]


def random_split(df, test_fraction=config.RANDOM_SPLIT_TEST_FRACTION, seed=config.RANDOM_SEED):
    """
    Random train/test split (for comparison only - NOT used to choose models).

    Output: (train_rows, test_rows).
    """
    test_rows = df.sample(frac=test_fraction, random_state=seed)
    return df.drop(test_rows.index), test_rows


def prepare_folds(raw_rows, fold_definitions):
    """
    Build the model-ready tables for every fold definition.

    Input:  raw labelled rows + list of fold definition dicts (see config.py).
    Output: list of Fold objects.
    """
    folds = []
    for definition in fold_definitions:
        train_rows, test_rows = split_by_dates(raw_rows, definition)
        folds.append(Fold(definition["name"], prepare_model_inputs(train_rows, test_rows)))
    return folds


# =============================================================================
# Metrics
# =============================================================================
def regression_metrics(actual, predicted):
    """
    The five accuracy measures we report (Spotter's metric is unknown), plus bias.

        MAE     - average absolute error in dollars (our main metric)
        RMSE    - like MAE but punishes big misses more
        MAPE    - average absolute error in % of the true price
        MedAPE  - median absolute % error (a "typical load" view)
        R2      - share of price variation explained (1.0 = perfect)
        Bias_%  - total predicted vs total actual (+ means we over-predict)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute_percentage_error = np.abs(error) / actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(100 * np.mean(absolute_percentage_error)),
        "MedAPE": float(100 * np.median(absolute_percentage_error)),
        "R2": float(1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "Bias_%": float(100 * (predicted.sum() / actual.sum() - 1)),
    }


def evaluate_predictions(test_table, predictions):
    """
    Score predictions on a labelled test table, twice:
      * "clean" - only rows whose label is NOT flagged as corrupted (the fair score)
      * "all"   - every row, corrupted labels included (what a naive score would show)

    Output: {"clean": metrics dict, "all": metrics dict}
    """
    actual = test_table["posted_rate"].to_numpy()
    is_clean = test_table["target_is_corrupt"].to_numpy() == 0
    return {"clean": regression_metrics(actual[is_clean], predictions[is_clean]),
            "all": regression_metrics(actual, predictions)}


def metrics_row(fold_name, scores):
    """Flatten {"clean": {...}, "all": {...}} into one row: clean_MAE, all_MAE, ..."""
    row = {"fold": fold_name}
    for subset, metrics in scores.items():
        for metric, value in metrics.items():
            row[f"{subset}_{metric}"] = value
    return row


# =============================================================================
# Cross-validation
# =============================================================================
@dataclass
class CVResult:
    """What one model spec achieved across the rolling folds."""
    spec: object
    fold_metrics: pd.DataFrame                          # one row per fold
    predictions: dict = field(default_factory=dict)     # fold name -> dollar predictions
    best_rounds: int = None                             # trees chosen by CV (boosting only)
    mean_training_rows: float = None                    # average fold training-set size
    rounds_curve: dict = field(default_factory=dict)    # trees -> mean clean MAE
    member_results: list = field(default_factory=list)  # ensembles: one CVResult per member

    def mean(self, metric="MAE", subset="clean"):
        """Average of a metric over the folds."""
        return float(self.fold_metrics[f"{subset}_{metric}"].mean())

    def std(self, metric="MAE", subset="clean"):
        """Standard deviation of a metric over the folds (how stable the model is)."""
        return float(self.fold_metrics[f"{subset}_{metric}"].std(ddof=0))


def cross_validate(spec, folds):
    """
    Train a fresh model on every fold's training rows and score it on the fold's test rows.

    Boosting models whose number of trees is not fixed are trained ONCE per fold
    with MAX_BOOSTING_ROUNDS trees and scored after every ROUND_CHECKPOINT; the
    checkpoint with the lowest mean MAE over all folds is kept (one value for
    every fold, so no fold gets its own hand-picked answer).

    Input:  ModelSpec + list of Fold objects.
    Output: CVResult.
    """
    choose_rounds = spec.is_boosting and spec.n_rounds is None
    checkpoints = [None]
    if choose_rounds:
        max_rounds = build_model(spec).n_rounds  # the model's own maximum (CatBoost has a lower one)
        checkpoints = [k for k in config.ROUND_CHECKPOINTS if k <= max_rounds]
    predictions_by_checkpoint = {checkpoint: {} for checkpoint in checkpoints}
    training_rows = []

    for fold in folds:
        model = build_model(spec)
        model.fit(fold.inputs.train_table)
        training_rows.append(len(select_training_rows(fold.inputs.train_table, spec.train_on_corrupt_rows)))
        for checkpoint in checkpoints:
            if checkpoint is None:
                fold_predictions = model.predict(fold.inputs.test_table)
            else:
                fold_predictions = model.predict(fold.inputs.test_table, n_rounds=checkpoint)
            predictions_by_checkpoint[checkpoint][fold.name] = fold_predictions

    # Mean clean MAE over the folds for every checkpoint -> keep the best one.
    rounds_curve = {}
    for checkpoint in checkpoints:
        fold_maes = [evaluate_predictions(fold.inputs.test_table,
                                          predictions_by_checkpoint[checkpoint][fold.name])["clean"]["MAE"]
                     for fold in folds]
        rounds_curve[checkpoint] = float(np.mean(fold_maes))
    best_checkpoint = min(rounds_curve, key=rounds_curve.get)
    predictions = predictions_by_checkpoint[best_checkpoint]

    rows = [metrics_row(fold.name, evaluate_predictions(fold.inputs.test_table, predictions[fold.name]))
            for fold in folds]
    best_rounds = best_checkpoint if choose_rounds else spec.n_rounds
    return CVResult(spec, pd.DataFrame(rows), predictions, best_rounds=best_rounds,
                    mean_training_rows=float(np.mean(training_rows)),
                    rounds_curve=rounds_curve if choose_rounds else {})


def combine_into_ensemble(spec, member_results, folds):
    """
    Score an average-of-members ensemble WITHOUT retraining anything: the members
    were already trained on exactly these folds, so averaging their stored
    predictions gives the same result as fitting the ensemble from scratch.
    """
    predictions = {fold.name: np.mean([result.predictions[fold.name] for result in member_results], axis=0)
                   for fold in folds}
    rows = [metrics_row(fold.name, evaluate_predictions(fold.inputs.test_table, predictions[fold.name]))
            for fold in folds]
    return CVResult(spec, pd.DataFrame(rows), predictions, member_results=list(member_results))


# =============================================================================
# Re-using the number of trees chosen by CV
# =============================================================================
def with_chosen_rounds(spec, cv_result, train_table):
    """
    Return a copy of spec whose boosting models use a FIXED number of trees:
    the CV choice, scaled by (rows in this training set / average fold training
    rows). More training data supports a few more trees.

    Works for single models and for ensembles (member by member).
    """
    if spec.model_type == "ensemble":
        members = [with_chosen_rounds(member_spec, member_result, train_table)
                   for member_spec, member_result in zip(spec.members, cv_result.member_results)]
        return spec.but(members=members)
    if not spec.is_boosting:
        return spec
    n_rows = len(select_training_rows(train_table, spec.train_on_corrupt_rows))
    n_rounds = max(50, round(cv_result.best_rounds * n_rows / cv_result.mean_training_rows))
    return spec.but(n_rounds=n_rounds)


def fit_and_evaluate(spec, cv_result, fold):
    """
    Train spec (with the CV-chosen number of trees) on a fold's training rows
    and score it on the fold's test rows. Used for the main holdout, the random
    split and the unseen-city check - no choice is made on these test rows.

    Output: (scores dict, predictions array, fitted model)
    """
    fixed_spec = with_chosen_rounds(spec, cv_result, fold.inputs.train_table)
    model = build_model(fixed_spec)
    model.fit(fold.inputs.train_table)
    predictions = model.predict(fold.inputs.test_table)
    return evaluate_predictions(fold.inputs.test_table, predictions), predictions, model
