"""
Run the whole freight-rate pipeline end-to-end with ONE command:

    python run_pipeline.py           # full run: every experiment (~70 minutes on a 4-core laptop)
    python run_pipeline.py --fast    # quick run (~10 minutes): re-uses the feature set and
                                     # settings stored in config.py

Phases
  0    Explore the data           -> facts + figures in reports/figures/
  1    Clean the data             -> one fix per data-quality issue, with counts
  2    Engineer features          -> FeatureBuilder (fitted on training rows only)
  3+4  Validate & compare models  -> time-based folds, outputs/model_comparison.csv
  5    Final model + predictions  -> validation_predictions.csv
  6    December chart inputs      -> outputs/december_chart_predictions.csv
  7    Run the provided scorer    -> scorer_results/candidate_december.png
"""

import argparse
import json
import random
import subprocess
import sys
import time

import numpy as np

from src import config
from src.data_cleaning import load_raw_data
from src.december import predict_december
from src.eda import run_eda
from src.experiments import run_experiments
from src.features import FEATURE_GROUPS, prepare_model_inputs
from src.predict import predict_validation, summarise_predictions, train_final_model


def banner(title):
    """Print a phase header."""
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


def phase_0_explore(train_raw, validation_raw):
    banner("PHASE 0 - EXPLORE THE DATA")
    return run_eda(train_raw, validation_raw)


def phase_1_and_2_clean_and_build_features(train_raw, validation_raw):
    """Clean both files and build the full-data feature tables (used by Phase 5)."""
    banner("PHASE 1 - DATA CLEANING (statistics learned from train-test.csv, applied to both files)")
    print("  Training file + validation file, in that order:")
    final_inputs = prepare_model_inputs(train_raw, validation_raw, verbose=True)
    train_table = final_inputs.train_table
    n_corrupt = int(train_table["target_is_corrupt"].sum())
    print(f"\n  Summary: {len(train_table):,} training rows -> {len(train_table) - n_corrupt:,} kept for training, "
          f"{n_corrupt} corrupted labels removed ({100 * n_corrupt / len(train_table):.2f}%); "
          f"{len(final_inputs.test_table):,} validation rows cleaned (none removed - every load needs a prediction)")

    banner("PHASE 2 - FEATURE ENGINEERING")
    n_features = sum(len(columns) for columns in FEATURE_GROUPS.values())
    print(f"  {n_features} engineered features in {len(FEATURE_GROUPS)} groups (the ablation in Phase 4 decides "
          f"which optional groups are kept):")
    for group, columns in FEATURE_GROUPS.items():
        print(f"    {group:<20s} ({len(columns)}): {', '.join(columns)}")
    validation_table = final_inputs.test_table
    train_cities = set(train_raw["pickup"]) | set(train_raw["delivery"])
    unseen = validation_table[~validation_table["pickup"].isin(train_cities)]
    levels = unseen.groupby("pickup")["pickup_city_level"].first()
    print("  Unseen validation cities get the price level of their 5 nearest training cities: "
          + ", ".join(f"{city} {100 * level:+.1f}%" for city, level in levels.items()))
    return final_inputs


def phase_3_and_4_experiments(train_raw, fast):
    banner("PHASE 3 + 4 - TIME-BASED VALIDATION AND MODEL COMPARISON")
    outcome = run_experiments(train_raw, fast=fast)
    log = outcome.log
    winner = log.results[outcome.winner_name]
    holdout = log.holdout_scores[outcome.winner_name]

    print("\n  Why it won:")
    print(f"    - lowest mean MAE over the 4 rolling time folds (Jul, Aug, Sep, Oct): ${winner.mean():.2f} "
          f"(std ${winner.std():.2f}), MAPE {winner.mean('MAPE'):.3f}%")
    baseline = log.results.get("A_baseline")
    if baseline is not None:
        print(f"    - {100 * (1 - winner.mean() / baseline.mean()):.0f}% lower error than the simple baseline "
              f"(${baseline.mean():.2f})")
    print(f"    - Sep-Oct holdout (trained on Jan-Aug only): MAE ${holdout['clean']['MAE']:.2f}, "
          f"MAPE {holdout['clean']['MAPE']:.3f}%, R2 {holdout['clean']['R2']:.4f}, "
          f"bias {holdout['clean']['Bias_%']:+.2f}%")
    print(f"    - scored on ALL rows (incl. corrupted labels) the same holdout MAE is "
          f"${holdout['all']['MAE']:.2f} - the corrupted labels add error no model can remove")

    random_scores = outcome.random_split_scores["clean"]
    print(f"\n  Random 80/20 split (for comparison only): MAE ${random_scores['MAE']:.2f}, "
          f"MAPE {random_scores['MAPE']:.3f}% - looks {100 * (1 - random_scores['MAE'] / holdout['clean']['MAE']):.0f}% "
          f"better than the honest time-based holdout, because random test loads share days (and the daily "
          f"price level) with training loads. Forecasting Nov-Dec never offers that.")

    unseen = outcome.unseen_city_scores
    print(f"\n  Unseen-city check ({', '.join(unseen['held_out_cities'])} removed from training): "
          f"{unseen['n_loads_touching']:,} Sep-Oct loads touch them")
    print(f"    MAPE on those loads {unseen['unseen']['MAPE']:.3f}% (vs {unseen['same_loads_when_seen']['MAPE']:.3f}% "
          f"when the cities were seen in training); other loads {unseen['other_loads']['MAPE']:.3f}%")

    print("\n  Permutation importance on the Sep-Oct holdout (rise in MAE when a feature group is shuffled):")
    for _, row in outcome.group_importance.iterrows():
        print(f"    {row['feature_group']:<20s} +${row['MAE_increase']:9.2f}  ({row['MAE_increase_%']:+8.1f}%)")
    print(f"\n  Saved {config.MODEL_COMPARISON_PATH.relative_to(config.PROJECT_ROOT)} and "
          f"{config.FOLD_DETAILS_PATH.relative_to(config.PROJECT_ROOT)}")
    return outcome


def phase_5_final_model(final_inputs, outcome):
    banner("PHASE 5 - FINAL TRAINING + VALIDATION PREDICTIONS")
    bundle = train_final_model(final_inputs, outcome)
    _submission, predictions = predict_validation(bundle, final_inputs.test_table)
    summary = summarise_predictions(final_inputs.train_table, final_inputs.test_table, predictions,
                                    final_inputs.cleaning_stats.reference_model.market_elasticity_)
    return bundle, summary


def phase_6_december(bundle, train_raw, validation_raw):
    banner("PHASE 6 - DECEMBER CHART PREDICTIONS")
    output, _model_rows, predictions = predict_december(bundle, train_raw, validation_raw)
    return {"min": float(predictions.min()), "max": float(predictions.max()), "mean": float(predictions.mean()),
            "first": float(predictions[0]), "last": float(predictions[-1]),
            "values": [float(value) for value in output["predicted_rate"]]}


def phase_7_scorer():
    banner("PHASE 7 - RUN THE PROVIDED SCORER (score.py)")
    command = [sys.executable, "score.py",
               "--predictions", config.VALIDATION_PREDICTIONS_PATH.name,
               "--december-predictions", str(config.DECEMBER_PREDICTIONS_PATH.relative_to(config.PROJECT_ROOT))]
    print("  $ python " + " ".join(command[1:]))
    completed = subprocess.run(command, cwd=config.PROJECT_ROOT, capture_output=True, text=True)
    print("  " + completed.stdout.strip().replace("\n", "\n  "))
    if completed.returncode != 0:
        print(completed.stderr)
        raise SystemExit("score.py rejected the outputs - see the message above")
    chart = config.SCORER_RESULTS_DIR / "candidate_december.png"
    if not chart.is_file():
        raise SystemExit(f"score.py finished but {chart} was not created")
    print(f"  Confirmed: {chart.relative_to(config.PROJECT_ROOT)} exists")


def main():
    parser = argparse.ArgumentParser(description="Freight rate prediction pipeline (Spotter assessment)")
    parser.add_argument("--fast", action="store_true",
                        help="skip the ablation / target / tuning experiments and CatBoost (~10 minutes)")
    args = parser.parse_args()

    started = time.time()
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)  # every model also gets the seed explicitly

    train_raw, validation_raw = load_raw_data()
    facts = phase_0_explore(train_raw, validation_raw)
    final_inputs = phase_1_and_2_clean_and_build_features(train_raw, validation_raw)
    outcome = phase_3_and_4_experiments(train_raw, args.fast)
    bundle, prediction_summary = phase_5_final_model(final_inputs, outcome)
    december_summary = phase_6_december(bundle, train_raw, validation_raw)
    phase_7_scorer()

    # A compact JSON record of the run (used to write the report).
    log = outcome.log
    winner = log.results[outcome.winner_name]
    run_summary = {
        "mode": "fast" if args.fast else "full",
        "runtime_minutes": round((time.time() - started) / 60, 1),
        "eda_facts": facts,
        "winner": outcome.winner_name,
        "winner_description": winner.spec.description,
        "winner_feature_groups": winner.spec.feature_groups,
        "dropped_feature_groups": outcome.dropped_groups,
        "final_model": str(bundle["spec"]),
        "cv_mean": {metric: winner.mean(metric) for metric in ["MAE", "RMSE", "MAPE", "MedAPE", "R2", "Bias_%"]},
        "cv_std_MAE": winner.std(),
        "cv_folds": winner.fold_metrics.to_dict(orient="records"),
        "holdout": log.holdout_scores[outcome.winner_name],
        "random_split": outcome.random_split_scores,
        "unseen_cities": outcome.unseen_city_scores,
        "feature_group_importance": outcome.group_importance.to_dict(orient="records"),
        "predictions": prediction_summary,
        "december": december_summary,
    }
    config.RUN_SUMMARY_PATH.write_text(json.dumps(run_summary, indent=2, default=str))
    banner(f"DONE in {run_summary['runtime_minutes']} minutes - summary saved to "
           f"{config.RUN_SUMMARY_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
