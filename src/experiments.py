"""
Phase 3 + 4 - run the experiments and pick the winning model.

Order of work (each step builds on the previous one):
  4a  Feature screen    - LightGBM with ALL feature groups, then with each
                          optional group switched off, then without every group
                          whose removal did not hurt. Keeps the best feature set.
  4b  Model comparison  - models A-E and G on that feature set (scored on the rolling
                          folds AND the Sep-Oct holdout), plus F = average of
                          the best 2 / best 3 models.
  4c  Ablations on the best single model - with/without quote_signal, with/without
                          the target encoding, and each target option
                          (log price / $ per mile / raw dollars).
  4d  Light tuning      - a small LightGBM grid on the best 4c variant, then the
                          ensembles are rebuilt if a model family improved.
  Winner                - lowest mean MAE over the 4 rolling folds (clean labels);
                          results within 0.2% count as a tie, broken by lower MAPE.

Extra checks on the winner (reported only, never used for choosing):
  * random split   - shows why a random split is too optimistic here
  * unseen cities  - 8 training cities removed from training; error measured on
                     the Sep-Oct loads that touch them
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config, plotting
from src.features import ALL_GROUPS, FEATURE_GROUPS, OPTIONAL_GROUPS, prepare_model_inputs
from src.models import ModelSpec
from src.validation import (
    Fold, combine_into_ensemble, cross_validate, evaluate_predictions, fit_and_evaluate, prepare_folds,
    random_split, regression_metrics, split_by_dates,
)


# =============================================================================
# Bookkeeping
# =============================================================================
class ExperimentLog:
    """Runs experiments, remembers every result, and prints one line per experiment."""

    def __init__(self, folds, holdout):
        self.folds = folds
        self.holdout = holdout
        self.results = {}           # experiment name -> CVResult
        self.stage = {}             # experiment name -> stage label
        self.holdout_scores = {}    # experiment name -> {"clean": ..., "all": ...}
        self.holdout_predictions = {}
        self._cache = {}            # recipe signature -> CVResult (avoid training twice)

    @staticmethod
    def _signature(spec):
        """Everything that changes the trained model, as a hashable value."""
        return (spec.model_type, tuple(sorted(spec.feature_groups)), spec.target_kind,
                tuple(sorted(spec.params.items())), spec.train_on_corrupt_rows, spec.n_rounds)

    def run(self, spec, stage, with_holdout=False):
        """Cross-validate one spec (re-using an identical earlier run if there is one)."""
        started = time.time()
        signature = self._signature(spec)
        if signature in self._cache:
            cached = self._cache[signature]
            result = type(cached)(**{**cached.__dict__, "spec": spec})
        else:
            result = cross_validate(spec, self.folds)
            self._cache[signature] = result
        self._store(spec, result, stage, with_holdout, started)
        return result

    def run_ensemble(self, spec, member_names, stage, with_holdout=False):
        """Score an ensemble from its members' stored fold predictions (no retraining)."""
        started = time.time()
        member_results = [self.results[name] for name in member_names]
        result = combine_into_ensemble(spec, member_results, self.folds)
        if with_holdout:
            for name in member_names:
                self.ensure_holdout(name)
            predictions = np.mean([self.holdout_predictions[name] for name in member_names], axis=0)
            self.holdout_predictions[spec.name] = predictions
            self.holdout_scores[spec.name] = evaluate_predictions(self.holdout.inputs.test_table, predictions)
        self._store(spec, result, stage, with_holdout=False, started=started)
        return result

    def ensure_holdout(self, name):
        """Score an experiment on the Sep-Oct holdout if that has not happened yet."""
        if name not in self.holdout_scores:
            result = self.results[name]
            scores, predictions, _model = fit_and_evaluate(result.spec, result, self.holdout)
            self.holdout_scores[name] = scores
            self.holdout_predictions[name] = predictions

    def _store(self, spec, result, stage, with_holdout, started):
        self.results[spec.name] = result
        self.stage[spec.name] = stage
        if with_holdout:
            self.ensure_holdout(spec.name)
        holdout_text = ""
        if spec.name in self.holdout_scores:
            holdout_text = f" | holdout MAE {self.holdout_scores[spec.name]['clean']['MAE']:7.2f}"
        rounds_text = f" | trees {result.best_rounds}" if result.best_rounds else ""
        print(f"    {spec.name:<58s} MAE {result.mean():7.2f} +/- {result.std():5.2f} "
              f"| MAPE {result.mean('MAPE'):5.3f}%{holdout_text}{rounds_text} "
              f"| {time.time() - started:5.0f}s", flush=True)

    def pick_best(self, names):
        """
        The selection rule: lowest mean MAE over the rolling folds; anything
        within TIE_TOLERANCE (0.2%) of the best counts as a tie, broken by MAPE.
        """
        mean_mae = {name: self.results[name].mean("MAE") for name in names}
        best_mae = min(mean_mae.values())
        tied = [name for name in names if mean_mae[name] <= best_mae * (1 + config.TIE_TOLERANCE)]
        return min(tied, key=lambda name: self.results[name].mean("MAPE"))

    def summary_table(self):
        """One row per experiment with the averaged metrics (saved as model_comparison.csv)."""
        rows = []
        for name, result in self.results.items():
            row = {"stage": self.stage[name], "experiment": name, "description": result.spec.description,
                   "model_type": result.spec.model_type, "target": result.spec.target_kind,
                   "feature_groups": " ".join(result.spec.feature_groups),
                   "params": str(result.spec.params) if result.spec.params else "",
                   "trees": result.best_rounds}
            for metric in ["MAE", "RMSE", "MAPE", "MedAPE", "R2", "Bias_%"]:
                row[f"cv_{metric}"] = result.mean(metric)
            row["cv_MAE_std"] = result.std("MAE")
            for _, fold_row in result.fold_metrics.iterrows():
                row[f"MAE_{fold_row['fold']}"] = fold_row["clean_MAE"]
            row["cv_MAE_all_rows"] = result.mean("MAE", subset="all")
            if name in self.holdout_scores:
                for metric in ["MAE", "RMSE", "MAPE", "R2", "Bias_%"]:
                    row[f"holdout_{metric}"] = self.holdout_scores[name]["clean"][metric]
                row["holdout_MAE_all_rows"] = self.holdout_scores[name]["all"]["MAE"]
            rows.append(row)
        return pd.DataFrame(rows)

    def fold_table(self):
        """Every fold's metrics for every experiment (saved as cv_fold_details.csv)."""
        tables = []
        for name, result in self.results.items():
            tables.append(result.fold_metrics.assign(experiment=name, stage=self.stage[name]))
        return pd.concat(tables, ignore_index=True)


# =============================================================================
# 4a - feature ablation
# =============================================================================
def run_feature_ablation(log):
    """
    LightGBM with every feature group, then each optional group switched off,
    then all "not helpful" groups switched off together.

    Output: (best feature groups, list of groups whose removal did not hurt)
    """
    print("  4a  Feature screen (LightGBM, log target): which optional feature groups help?")
    reference = ModelSpec("C_lightgbm | all features", "lightgbm", list(ALL_GROUPS),
                          description="LightGBM with every feature group")
    reference_mae = log.run(reference, stage="4a feature screen").mean()

    not_helpful = []
    candidates = [reference.name]
    for group in OPTIONAL_GROUPS:
        spec = reference.but(name=f"C_lightgbm | without {group}",
                             feature_groups=[g for g in ALL_GROUPS if g != group],
                             description=f"all features except '{group}'")
        if log.run(spec, stage="4a feature screen").mean() <= reference_mae:
            not_helpful.append(group)
        candidates.append(spec.name)

    if len(not_helpful) >= 2:
        reduced = reference.but(name="C_lightgbm | without " + " + ".join(not_helpful),
                                feature_groups=[g for g in ALL_GROUPS if g not in not_helpful],
                                description="all features except every group whose removal did not hurt")
        log.run(reduced, stage="4a feature screen")
        candidates.append(reduced.name)

    best = log.pick_best(candidates)
    best_groups = log.results[best].spec.feature_groups
    dropped = [g for g in ALL_GROUPS if g not in best_groups]
    print(f"      -> best feature set: '{best}' (dropped: {', '.join(dropped) or 'nothing'})")
    return best_groups, not_helpful


# =============================================================================
# 4b - model comparison (+ ensembles)
# =============================================================================
def model_comparison_specs(feature_groups, include_catboost=True):
    """The single-model recipes A-E and G, all on the same feature set."""
    specs = [
        ModelSpec("A_baseline", "baseline",
                  description="median $/mile by equipment & distance band, scaled by market_index"),
        ModelSpec("B_ridge", "ridge", feature_groups, description="ridge regression on log(price)"),
        ModelSpec("C_lightgbm", "lightgbm", feature_groups, description="LightGBM on log(price), L2 loss"),
        ModelSpec("D_lightgbm_huber", "lightgbm_huber", feature_groups, train_on_corrupt_rows=True,
                  description="LightGBM on log(price), Huber loss, corrupt labels KEPT in training"),
    ]
    if include_catboost:
        specs.append(ModelSpec("E_catboost", "catboost", feature_groups,
                               description="CatBoost on log(price) + raw city/equipment categories"))
    specs.append(ModelSpec("G_ridge_then_lightgbm", "ridge_lightgbm", feature_groups,
                           description="Ridge on log(price), then LightGBM learns Ridge's errors "
                                       "(the time trend stays in the linear part)"))
    return specs


def add_ensembles(log, candidate_names, stage, suffix=""):
    """Build F = average of the best 2 and best 3 candidates (different model families)."""
    # Keep only the best experiment of each model family, so the ensemble mixes families.
    best_by_family = {}
    for name in candidate_names:
        family = log.results[name].spec.model_type
        if family not in best_by_family or log.results[name].mean() < log.results[best_by_family[family]].mean():
            best_by_family[family] = name
    ranked = sorted(best_by_family.values(), key=lambda name: log.results[name].mean())
    existing_member_sets = {tuple(sorted(member.name for member in result.spec.members))
                            for result in log.results.values() if result.spec.model_type == "ensemble"}
    names = []
    for size in (2, 3):
        if len(ranked) < size:
            continue
        members = ranked[:size]
        if tuple(sorted(members)) in existing_member_sets:
            continue  # this exact ensemble was already scored
        spec = ModelSpec(f"F_ensemble_top{size}{suffix}", "ensemble",
                         members=[log.results[name].spec for name in members],
                         description="average of " + " + ".join(members))
        log.run_ensemble(spec, members, stage=stage, with_holdout=True)
        names.append(spec.name)
    return names


def run_model_comparison(log, feature_groups, include_catboost=True):
    """Models A-E on the chosen features, then the ensembles. Output: experiment names."""
    print("  4b  Model comparison (rolling folds + Sep-Oct holdout)")
    names = []
    for spec in model_comparison_specs(feature_groups, include_catboost):
        log.run(spec, stage="4b model comparison", with_holdout=True)
        names.append(spec.name)
    single_models = [name for name in names if name != "A_baseline"]
    names += add_ensembles(log, single_models, stage="4b model comparison")
    return names


# =============================================================================
# 4c / 4d - ablations on the best model, and light tuning
# =============================================================================
def run_best_model_ablations(log, base_name):
    """
    The ablations the brief asks for, run on the best single model of 4b:
      * with / without quote_signal
      * with / without the target encoding (city_level group)
      * each target option: log (the default), $ per mile, raw dollars
    Output: experiment names (including the base).
    """
    print(f"  4c  Ablations on the best single model ({base_name})")
    base = log.results[base_name].spec
    names = [base_name]
    for group, label in [("quote_signal", "quote_signal"), ("city_level", "target encoding (city_level)")]:
        if group in base.feature_groups:
            groups = [g for g in base.feature_groups if g != group]
            name, description = f"{base_name} | without {group}", f"{base.description}; without {label}"
        else:
            groups = [g for g in ALL_GROUPS if g in base.feature_groups or g == group]
            name, description = f"{base_name} | with {group}", f"{base.description}; {label} added back"
        log.run(base.but(name=name, feature_groups=groups, description=description), stage="4c ablation on best")
        names.append(name)
    for kind, label in [("per_mile", "$ per mile"), ("raw", "raw dollars")]:
        spec = base.but(name=f"{base_name} | target={kind}", target_kind=kind,
                        description=f"{base.description}; target = {label}")
        log.run(spec, stage="4c ablation on best")
        names.append(spec.name)
    return names


def run_tuning(log, base_name):
    """A small LightGBM grid around the current best recipe. Output: experiment names."""
    base = log.results[base_name].spec
    if base.model_type not in ("lightgbm", "lightgbm_huber", "ridge_lightgbm"):
        print(f"  4d  Tuning skipped ({base_name} has no LightGBM part)")
        return [base_name]
    print(f"  4d  Light tuning (on {base_name})")
    names = [base_name]
    for grid_point in config.LIGHTGBM_TUNING_GRID:
        label = ", ".join(f"{key}={value}" for key, value in grid_point.items())
        spec = base.but(name=f"{base_name} | {label}", params={**base.params, **grid_point},
                        description=f"{base.description}; {label}")
        log.run(spec, stage="4d tuning")
        names.append(spec.name)
    return names


# =============================================================================
# Extra checks on the winner
# =============================================================================
def random_split_check(log, winner_name, train_raw):
    """Train/test on a RANDOM 80/20 split, to compare with the time-based scores."""
    train_rows, test_rows = random_split(train_raw)
    fold = Fold("random 80/20", prepare_model_inputs(train_rows, test_rows))
    result = log.results[winner_name]
    scores, _predictions, _model = fit_and_evaluate(result.spec, result, fold)
    return scores


def unseen_city_check(log, winner_name, train_raw):
    """
    Simulate the 8 validation-only cities: remove 8 random training cities from the
    Jan-Aug training rows, train, and score the Sep-Oct loads that touch them.
    The same loads scored by the normal holdout model (cities seen) give the
    fair comparison.
    """
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    all_cities = sorted(set(train_raw["pickup"]) | set(train_raw["delivery"]))
    held_out = list(random_generator.choice(all_cities, size=config.N_SIMULATED_UNSEEN_CITIES, replace=False))

    train_rows, test_rows = split_by_dates(train_raw, config.MAIN_HOLDOUT)
    touches = train_rows["pickup"].isin(held_out) | train_rows["delivery"].isin(held_out)
    fold = Fold("unseen cities", prepare_model_inputs(train_rows[~touches], test_rows))
    result = log.results[winner_name]
    _scores, predictions, _model = fit_and_evaluate(result.spec, result, fold)

    test_table = fold.inputs.test_table
    is_clean = test_table["target_is_corrupt"].to_numpy() == 0
    touches_unseen = (test_table["pickup"].isin(held_out) | test_table["delivery"].isin(held_out)).to_numpy()

    # Same rows from the normal holdout run, where those cities WERE in training.
    log.ensure_holdout(winner_name)
    seen_predictions = pd.Series(log.holdout_predictions[winner_name],
                                 index=log.holdout.inputs.test_table["load_id"].to_numpy())
    seen_predictions = seen_predictions.reindex(test_table["load_id"]).to_numpy()

    actual = test_table["posted_rate"].to_numpy()
    rows_unseen = is_clean & touches_unseen
    rows_other = is_clean & ~touches_unseen
    return {
        "held_out_cities": held_out,
        "n_loads_touching": int(rows_unseen.sum()),
        "unseen": regression_metrics(actual[rows_unseen], predictions[rows_unseen]),
        "same_loads_when_seen": regression_metrics(actual[rows_unseen], seen_predictions[rows_unseen]),
        "other_loads": regression_metrics(actual[rows_other], predictions[rows_other]),
    }


def feature_groups_used(spec):
    """The feature groups a spec uses (for an ensemble: the union over its members)."""
    if spec.model_type == "ensemble":
        groups = set()
        for member in spec.members:
            groups |= set(feature_groups_used(member))
        return [g for g in ALL_GROUPS if g in groups]
    return list(spec.feature_groups)


def permutation_importance_check(log, winner_name):
    """
    Model-agnostic feature importance on the Sep-Oct holdout.

    Shuffle one feature GROUP at a time (the same row shuffle for every column in
    the group, so the group keeps its internal consistency) and measure how much
    the MAE on clean labels rises. A big rise means the model relies on that
    group. Works for any model - Ridge, trees, the hybrid or an ensemble.

    Output: DataFrame with one row per feature group.
    """
    result = log.results[winner_name]
    _scores, _predictions, model = fit_and_evaluate(result.spec, result, log.holdout)
    test_table = log.holdout.inputs.test_table
    clean = test_table[test_table["target_is_corrupt"] == 0]
    actual = clean["posted_rate"].to_numpy()
    base_mae = regression_metrics(actual, model.predict(clean))["MAE"]

    random_generator = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for group in feature_groups_used(result.spec):
        new_order = random_generator.permutation(len(clean))
        shuffled = clean.copy()
        for column in FEATURE_GROUPS[group]:
            moved = clean[column].iloc[new_order]
            moved.index = clean.index  # keep the column's type (e.g. categories), only move the values
            shuffled[column] = moved
        shuffled_mae = regression_metrics(actual, model.predict(shuffled))["MAE"]
        rows.append({"feature_group": group, "n_columns": len(FEATURE_GROUPS[group]),
                     "holdout_MAE": base_mae, "MAE_when_shuffled": shuffled_mae,
                     "MAE_increase": shuffled_mae - base_mae,
                     "MAE_increase_%": 100 * (shuffled_mae / base_mae - 1)})
    return pd.DataFrame(rows).sort_values("MAE_increase", ascending=False).reset_index(drop=True)


# =============================================================================
# The whole of Phase 3 + 4
# =============================================================================
@dataclass
class ExperimentOutcome:
    """Everything later phases need from the experiments."""
    winner_name: str
    log: ExperimentLog
    random_split_scores: dict
    unseen_city_scores: dict
    group_importance: pd.DataFrame
    dropped_groups: list


def run_experiments(train_raw, fast=False):
    """
    Run Phase 3 (validation design) and Phase 4 (model comparison).

    Input:  raw labelled training rows; fast=True skips 4a, 4c, 4d and CatBoost
            and uses the choices stored in config.py instead.
    Output: ExperimentOutcome.
    """
    started = time.time()
    print("  Preparing the 4 rolling folds and the Sep-Oct holdout "
          "(cleaning + features fitted on each fold's own training rows) ...")
    folds = prepare_folds(train_raw, config.ROLLING_FOLDS)
    holdout = prepare_folds(train_raw, [config.MAIN_HOLDOUT])[0]
    for fold in folds + [holdout]:
        train_table, test_table = fold.inputs.train_table, fold.inputs.test_table
        train_cities = set(train_table["pickup"]) | set(train_table["delivery"])
        n_unseen = int((~test_table["pickup"].isin(train_cities) | ~test_table["delivery"].isin(train_cities)).sum())
        print(f"    {fold.name:<16s} train {train_table['date'].min():%b %d}-{train_table['date'].max():%b %d} "
              f"({len(train_table):,} rows, {int(train_table['target_is_corrupt'].sum())} corrupt) -> "
              f"test {test_table['date'].min():%b %d}-{test_table['date'].max():%b %d} ({len(test_table):,} rows, "
              f"{int(test_table['target_is_corrupt'].sum())} corrupt, {n_unseen} with an unseen city)")
    log = ExperimentLog(folds, holdout)

    if fast:
        print("  --fast: skipping the feature ablation, target options, tuning and CatBoost")
        feature_groups = list(config.FAST_MODE_FEATURE_GROUPS)
        dropped = [g for g in ALL_GROUPS if g not in feature_groups]
        candidates = run_model_comparison(log, feature_groups, include_catboost=False)
        tuning = config.FAST_MODE_TUNING
        if tuning:
            base = log.results[tuning["base"]].spec
            label = ", ".join(f"{key}={value}" for key, value in tuning["params"].items())
            tuned = base.but(name=f"{base.name} | {label}", params={**base.params, **tuning["params"]},
                             target_kind=tuning.get("target_kind", base.target_kind),
                             description=f"{base.description}; settings from the last full run")
            log.run(tuned, stage="4d tuning (from config)")
            candidates.append(tuned.name)
            singles = [n for n in candidates if n != "A_baseline" and not n.startswith("F_")]
            candidates += add_ensembles(log, singles, stage="4d tuning (from config)", suffix="_tuned")
    else:
        feature_groups, _not_helpful = run_feature_ablation(log)
        dropped = [g for g in ALL_GROUPS if g not in feature_groups]
        candidates = run_model_comparison(log, feature_groups)
        single_models = [n for n in candidates if n != "A_baseline" and not n.startswith("F_")]
        best_single = log.pick_best(single_models)
        ablation_names = run_best_model_ablations(log, best_single)
        tuning_names = run_tuning(log, log.pick_best(ablation_names))
        candidates += ablation_names + tuning_names
        # If 4c/4d improved a model family, rebuild the ensembles with the improved member
        # (add_ensembles skips any ensemble that was already scored).
        singles = [n for n in candidates if n != "A_baseline" and not n.startswith("F_")]
        candidates += add_ensembles(log, singles, stage="4d tuning", suffix="_tuned")

    candidates = list(dict.fromkeys(candidates))  # unique, order kept
    winner = log.pick_best(candidates)
    log.ensure_holdout(winner)
    print(f"\n  Winner: {winner}  (mean MAE {log.results[winner].mean():.2f} over the rolling folds, "
          f"MAPE {log.results[winner].mean('MAPE'):.3f}%)")

    print("  Extra checks on the winner ...")
    random_scores = random_split_check(log, winner, train_raw)
    unseen_scores = unseen_city_check(log, winner, train_raw)
    group_importance = permutation_importance_check(log, winner)

    # Save tables + figures.
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = log.summary_table()
    summary.to_csv(config.MODEL_COMPARISON_PATH, index=False)
    log.fold_table().to_csv(config.FOLD_DETAILS_PATH, index=False)
    group_importance.to_csv(config.GROUP_IMPORTANCE_PATH, index=False)
    plotting.plot_group_importance(group_importance, winner)
    comparison = summary[summary["stage"].str.startswith("4b")]
    plotting.plot_model_comparison(list(comparison["experiment"]), comparison["cv_MAE"].to_numpy(),
                                   comparison["cv_MAE_std"].to_numpy(),
                                   "Model comparison - time-based rolling folds", "model_comparison.png")
    ablation = summary[summary["stage"].str.startswith("4a")]
    if len(ablation):
        plotting.plot_model_comparison(list(ablation["experiment"]), ablation["cv_MAE"].to_numpy(),
                                       ablation["cv_MAE_std"].to_numpy(),
                                       "Feature screen - LightGBM, time-based rolling folds",
                                       "feature_screen.png")
    print(f"  Phase 3+4 finished in {(time.time() - started) / 60:.1f} minutes")
    return ExperimentOutcome(winner, log, random_scores, unseen_scores, group_importance, dropped)
