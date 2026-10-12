"""
Phase 5 - final training and the validation predictions.

The winning recipe is retrained on ALL labelled Jan-Oct rows (cleaning and
features re-fitted on the full training set) with the number of trees found
during cross-validation, scaled up for the larger training set. It then
predicts every load in validation.csv, fills the provided template and saves
validation_predictions.csv after a set of sanity checks.
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import config, plotting
from src.models import build_model, select_training_rows
from src.validation import with_chosen_rounds


def describe_trees(spec):
    """Human-readable number of trees, e.g. 'C_lightgbm: 3,100 trees'."""
    if spec.model_type == "ensemble":
        return "; ".join(describe_trees(member) for member in spec.members)
    if spec.is_boosting:
        return f"{spec.name}: {spec.n_rounds:,} trees"
    return f"{spec.name}: no trees"


def train_final_model(final_inputs, outcome):
    """
    Retrain the winner on every labelled row and save it.

    Input:  ModelInputs for (all training rows, validation rows) + ExperimentOutcome.
    Output: bundle dict with the fitted model and everything needed to process new rows.
    """
    result = outcome.log.results[outcome.winner_name]
    final_spec = with_chosen_rounds(result.spec, result, final_inputs.train_table)
    model = build_model(final_spec)
    model.fit(final_inputs.train_table)

    n_clean_rows = len(select_training_rows(final_inputs.train_table, keep_corrupt_rows=False))
    n_all_rows = len(final_inputs.train_table)
    bundle = {
        "spec": final_spec,
        "model": model,
        "cleaning_stats": final_inputs.cleaning_stats,
        "feature_builder": final_inputs.feature_builder,
        "n_clean_training_rows": n_clean_rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.FINAL_MODEL_PATH)
    print(f"    - trained '{outcome.winner_name}' on all Jan-Oct data: {n_clean_rows:,} clean rows "
          f"(of {n_all_rows:,}; corrupt labels are only used by Huber-loss models)")
    print(f"    - number of trees, scaled up from the CV choice for the larger training set: "
          f"{describe_trees(final_spec)}")
    print(f"    - saved the model bundle to {config.FINAL_MODEL_PATH.relative_to(config.PROJECT_ROOT)}")

    if hasattr(model, "feature_importance"):
        plotting.plot_feature_importance(model.feature_importance(),
                                         f"Tree part of the final model - gain per feature ({outcome.winner_name})")
    return bundle


def build_submission(validation_table, predictions):
    """
    Fill the provided template with our predictions (template order is kept).

    Output: DataFrame with exactly two columns: load_id, predicted_rate.
    """
    template = pd.read_csv(config.TEMPLATE_PATH)
    prediction_by_id = pd.Series(predictions, index=validation_table["load_id"].to_numpy())
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].map(prediction_by_id).round(2)
    return submission


def check_submission(submission):
    """
    The same checks score.py runs, done BEFORE saving so problems surface early.
    Raises ValueError if anything is wrong.
    """
    expected_ids = [f"TE-{number:06d}" for number in range(1, 12_001)]
    problems = []
    if list(submission.columns) != ["load_id", "predicted_rate"]:
        problems.append("columns must be exactly load_id,predicted_rate")
    if len(submission) != 12_000:
        problems.append(f"expected 12,000 rows, found {len(submission):,}")
    if sorted(submission["load_id"].astype(str)) != expected_ids:
        problems.append("load_id values do not match TE-000001..TE-012000")
    if submission["predicted_rate"].isna().any():
        problems.append("some predictions are missing")
    if not np.isfinite(submission["predicted_rate"]).all():
        problems.append("some predictions are not finite")
    if (submission["predicted_rate"] <= 0).any():
        problems.append("some predictions are not positive")
    if problems:
        raise ValueError("submission failed its checks: " + "; ".join(problems))
    print("    - checks passed: 12,000 rows, ids TE-000001..TE-012000, no missing values, all > 0")


def save_submission(submission):
    """Write validation_predictions.csv to the project root, as the brief asks."""
    submission.to_csv(config.VALIDATION_PREDICTIONS_PATH, index=False)
    print(f"    - saved {config.VALIDATION_PREDICTIONS_PATH.name} (project root)")


def summarise_predictions(train_table, validation_table, predictions, market_elasticity):
    """
    Print summary statistics and compare $/mile with the training period.

    Input:  training / validation feature tables, the predictions, and the
            market elasticity learned by the reference price model.
    Output: dict of summary numbers (for the run summary / report).
    """
    clean_train = train_table[train_table["target_is_corrupt"] == 0]
    train_rpm = clean_train["posted_rate"] / clean_train["distance"]
    predicted_rpm = pd.Series(predictions / validation_table["distance"].to_numpy(),
                              index=validation_table.index)

    train_monthly = train_rpm.groupby(clean_train["date"].dt.to_period("M")).mean()
    predicted_monthly = predicted_rpm.groupby(validation_table["date"].dt.to_period("M")).mean()
    summary = {
        "prediction_min": float(np.min(predictions)),
        "prediction_median": float(np.median(predictions)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_max": float(np.max(predictions)),
        "rpm_train_mean": float(train_rpm.mean()),
        "rpm_october_mean": float(train_monthly.iloc[-1]),
        "rpm_predicted_mean": float(predicted_rpm.mean()),
        "rpm_predicted_by_month": {str(month): float(value) for month, value in predicted_monthly.items()},
        "market_train_mean": float(clean_train["market_index"].mean()),
        "market_validation_mean": float(validation_table["market_index"].mean()),
    }
    print(f"    - predicted rate: min ${summary['prediction_min']:,.0f}, median ${summary['prediction_median']:,.0f}, "
          f"mean ${summary['prediction_mean']:,.0f}, max ${summary['prediction_max']:,.0f}")
    print(f"    - average $/mile: training Jan-Oct ${summary['rpm_train_mean']:.3f}, October "
          f"${summary['rpm_october_mean']:.3f}, predicted Nov ${predicted_monthly.iloc[0]:.3f}, "
          f"Dec ${predicted_monthly.iloc[-1]:.3f}")
    market_effect = 100 * market_elasticity * np.log(summary["market_validation_mean"] / summary["market_train_mean"])
    print(f"    - market_index: training mean {summary['market_train_mean']:.3f} vs validation "
          f"{summary['market_validation_mean']:.3f}; with an elasticity of {market_elasticity:.2f} that alone "
          f"moves prices by only {market_effect:+.1f}% vs the training average. The calendar level carried "
          f"over from October (+ the December quarter-end ramp) pushes the other way, so Nov/Dec $/mile "
          f"should be compared with October rather than with the Jan-Oct average.")

    plotting.plot_monthly_rate_per_mile(train_monthly.to_timestamp(), predicted_monthly.to_timestamp())
    return summary


def predict_validation(bundle, validation_table):
    """
    Predict every validation load, check, save and summarise.

    Output: (submission DataFrame, numpy array of predictions)
    """
    predictions = bundle["model"].predict(validation_table)
    submission = build_submission(validation_table, predictions)
    check_submission(submission)
    save_submission(submission)
    return submission, predictions
