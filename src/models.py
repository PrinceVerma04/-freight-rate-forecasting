"""
Phase 4 - the model zoo.

Every model has the same small interface, so they can be swapped freely:

    model = build_model(spec)
    model.fit(train_table)               # table from FeatureBuilder (with posted_rate)
    dollars = model.predict(any_table)   # predicted posted_rate in dollars

    A  BaselineModel           - median $/mile by equipment & distance, scaled by market_index
    B  RidgeModel              - linear regression with an L2 penalty, on log(price)
    C  LightGBMModel           - gradient-boosted trees on log(price), L2 loss
    D  LightGBMModel           - same, Huber loss, trained WITH the suspected corrupt rows
    E  CatBoostModel           - gradient-boosted trees with native categorical handling
    F  EnsembleModel           - average of the best 2-3 models
    G  RidgeThenLightGBMModel  - LightGBM that starts from the Ridge prediction and only
                                 learns its corrections (added after B and C turned out
                                 to make very different, complementary errors)

All models drop the flagged corrupt labels from their training rows, except
model D, whose whole point is to keep them and rely on a robust loss instead.

Boosting models (C, D, E, G) build trees one after another, each fixing the
errors of the previous ones. `predict(table, n_rounds=k)` uses only the first
k trees, which lets cross-validation find the best number of trees from a
single training run per fold (see validation.py).
"""

from dataclasses import dataclass, field, replace

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data_cleaning import ReferencePriceModel
from src.features import (
    CATEGORICAL_FEATURES, FEATURE_GROUPS, columns_for_groups, from_model_target, to_model_target,
)

BOOSTING_MODEL_TYPES = ["lightgbm", "lightgbm_huber", "catboost", "ridge_lightgbm"]


# =============================================================================
# A model "recipe"
# =============================================================================
@dataclass
class ModelSpec:
    """
    Everything needed to build one model. Experiments are just lists of specs.

    name                  - label used in the results table
    model_type            - "baseline", "ridge", "lightgbm", "lightgbm_huber",
                            "catboost", "ridge_lightgbm" or "ensemble"
    feature_groups        - which FEATURE_GROUPS the model may use
    target_kind           - "log", "per_mile" or "raw" (see features.py)
    params                - overrides of the default hyper-parameters
    train_on_corrupt_rows - keep the flagged corrupt labels in training? (model D)
    members               - the member specs (ensembles only)
    n_rounds              - number of trees (boosting models). None means
                            "train MAX_BOOSTING_ROUNDS and let CV choose".
    description           - one line for the results table
    """
    name: str
    model_type: str
    feature_groups: list = field(default_factory=list)
    target_kind: str = "log"
    params: dict = field(default_factory=dict)
    train_on_corrupt_rows: bool = False
    members: list = field(default_factory=list)
    n_rounds: int = None
    description: str = ""

    def but(self, **changes):
        """Return a copy of this spec with some fields changed (handy for ablations)."""
        return replace(self, **changes)

    @property
    def is_boosting(self):
        return self.model_type in BOOSTING_MODEL_TYPES


def select_training_rows(train_table, keep_corrupt_rows):
    """Drop the flagged corrupt labels, unless the model is meant to keep them."""
    if keep_corrupt_rows:
        return train_table
    return train_table[train_table["target_is_corrupt"] == 0]


# =============================================================================
# A - baseline
# =============================================================================
class BaselineModel:
    """
    Model A: the reference price from Phase 1 used directly as a forecast -
    median $/mile per equipment type and distance band, scaled by market_index.
    Any useful model must beat this.
    """

    def __init__(self, train_on_corrupt_rows=False):
        self.train_on_corrupt_rows = train_on_corrupt_rows

    def fit(self, train_table):
        """Learn the $/mile curves and market elasticity from the training rows."""
        rows = select_training_rows(train_table, self.train_on_corrupt_rows)
        self.reference_ = ReferencePriceModel().fit(rows)
        return self

    def predict(self, table):
        """Expected price in dollars."""
        return self.reference_.predict(table)


# =============================================================================
# B - ridge regression
# =============================================================================
class RidgeModel:
    """
    Model B: a linear model on log(price), with numeric features scaled and
    region labels one-hot encoded. Ridge adds a small penalty on large
    coefficients, which keeps the many correlated features stable.

    Note: unlike a tree model, a linear model EXTRAPOLATES months_since_start,
    i.e. it continues the 2025 trend into Nov/Dec instead of holding it flat.
    """

    def __init__(self, feature_columns, target_kind="log", alpha=config.RIDGE_ALPHA,
                 train_on_corrupt_rows=False):
        self.feature_columns = list(feature_columns)
        self.target_kind = target_kind
        self.alpha = alpha
        self.train_on_corrupt_rows = train_on_corrupt_rows

    def _matrix(self, table):
        matrix = table[self.feature_columns].copy()
        for column in CATEGORICAL_FEATURES:
            if column in matrix.columns:
                matrix[column] = matrix[column].astype(int)
        return matrix

    def fit(self, train_table):
        """Fit gap-filler + scaler + one-hot encoder + ridge regression."""
        rows = select_training_rows(train_table, self.train_on_corrupt_rows)
        categorical = [column for column in self.feature_columns if column in CATEGORICAL_FEATURES]
        numeric = [column for column in self.feature_columns if column not in CATEGORICAL_FEATURES]
        preprocess = ColumnTransformer([
            ("numeric", Pipeline([("fill_gaps", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        self.pipeline_ = Pipeline([("prepare", preprocess), ("ridge", Ridge(alpha=self.alpha))])
        target = to_model_target(rows["posted_rate"], rows["distance"], self.target_kind)
        self.pipeline_.fit(self._matrix(rows), target)
        return self

    def predict_target_space(self, table):
        """The raw model output (e.g. log price), before turning it back into dollars."""
        return self.pipeline_.predict(self._matrix(table))

    def predict(self, table):
        """Predicted price in dollars."""
        return from_model_target(self.predict_target_space(table), table["distance"], self.target_kind)


# =============================================================================
# C / D - LightGBM
# =============================================================================
class LightGBMModel:
    """
    Models C and D: LightGBM gradient-boosted decision trees.

    Trees capture curved effects (distance, weight) and interactions without
    hand-made formulas. objective="huber" (model D) makes large errors count
    linearly instead of squared, so corrupted labels barely pull the model.
    """

    def __init__(self, feature_columns, target_kind="log", params=None, objective="regression",
                 n_rounds=None, train_on_corrupt_rows=False):
        self.feature_columns = list(feature_columns)
        self.target_kind = target_kind
        self.params = {**config.LIGHTGBM_PARAMS, **(params or {})}
        if objective == "huber":
            self.params["objective"] = "huber"
            self.params["alpha"] = config.HUBER_DELTA  # LightGBM calls the Huber delta "alpha"
        self.n_rounds = n_rounds or config.MAX_BOOSTING_ROUNDS
        self.train_on_corrupt_rows = train_on_corrupt_rows

    def fit(self, train_table):
        """Train n_rounds trees on the training rows."""
        rows = select_training_rows(train_table, self.train_on_corrupt_rows)
        target = to_model_target(rows["posted_rate"], rows["distance"], self.target_kind)
        dataset = lgb.Dataset(rows[self.feature_columns], target)
        self.booster_ = lgb.train(self.params, dataset, num_boost_round=self.n_rounds)
        self.n_training_rows_ = len(rows)
        return self

    def predict(self, table, n_rounds=None):
        """Predicted price in dollars, using the first n_rounds trees (all if None)."""
        raw_prediction = self.booster_.predict(table[self.feature_columns], num_iteration=n_rounds)
        return from_model_target(raw_prediction, table["distance"], self.target_kind)

    def feature_importance(self):
        """Total gain (error reduction) contributed by each feature, as a Series."""
        gains = self.booster_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_columns).sort_values(ascending=False)


# =============================================================================
# E - CatBoost
# =============================================================================
class CatBoostModel:
    """
    Model E: CatBoost gradient-boosted trees.

    CatBoost's speciality is text categories: it turns city names into numbers
    internally with its own leakage-safe ("ordered") target statistics. So on
    top of the shared features it also receives the raw pickup / delivery /
    equipment labels.
    """

    RAW_LABEL_COLUMNS = ["pickup", "delivery", "equipment"]

    def __init__(self, feature_columns, target_kind="log", params=None, n_rounds=None,
                 train_on_corrupt_rows=False):
        self.columns = list(feature_columns) + [c for c in self.RAW_LABEL_COLUMNS if c not in feature_columns]
        self.categorical_columns = [c for c in self.columns
                                    if c in self.RAW_LABEL_COLUMNS + CATEGORICAL_FEATURES]
        self.target_kind = target_kind
        self.params = {**config.CATBOOST_PARAMS, **(params or {})}
        self.n_rounds = n_rounds or config.CATBOOST_MAX_ROUNDS
        self.train_on_corrupt_rows = train_on_corrupt_rows

    def _pool(self, table, target=None):
        matrix = table[self.columns].copy()
        for column in self.categorical_columns:
            matrix[column] = matrix[column].astype(str)  # CatBoost wants text or integers
        return Pool(matrix, label=target, cat_features=self.categorical_columns)

    def fit(self, train_table):
        """Train n_rounds trees on the training rows."""
        rows = select_training_rows(train_table, self.train_on_corrupt_rows)
        target = to_model_target(rows["posted_rate"], rows["distance"], self.target_kind)
        self.model_ = CatBoostRegressor(iterations=self.n_rounds, **self.params)
        self.model_.fit(self._pool(rows, target))
        self.n_training_rows_ = len(rows)
        return self

    def predict(self, table, n_rounds=None):
        """Predicted price in dollars, using the first n_rounds trees (all if None)."""
        raw_prediction = self.model_.predict(self._pool(table), ntree_end=n_rounds or 0)
        return from_model_target(raw_prediction, table["distance"], self.target_kind)

    def feature_importance(self):
        """CatBoost's own importance score per column, as a Series."""
        scores = self.model_.get_feature_importance()
        return pd.Series(scores, index=self.columns).sort_values(ascending=False)


# =============================================================================
# G - Ridge first, then LightGBM on what Ridge missed
# =============================================================================
class RidgeThenLightGBMModel:
    """
    Model G: a two-step model.

      1. Ridge regression (model B) learns the smooth, mostly multiplicative
         structure of prices on the log scale - including the 2025 price-level
         trend, which a linear model can carry forward into Nov/Dec.
      2. LightGBM then starts FROM Ridge's prediction (LightGBM's "init_score")
         and only learns the corrections Ridge cannot express: curved weight and
         distance effects, city details, interactions. Its trees do NOT see
         months_since_start, so the time trend stays the job of the linear part
         (trees could only freeze it at the last training month).

    Final prediction = Ridge output + LightGBM correction, turned back into dollars.
    """

    TREND_COLUMNS = FEATURE_GROUPS["time_trend"]

    def __init__(self, feature_columns, target_kind="log", params=None, n_rounds=None,
                 train_on_corrupt_rows=False):
        self.ridge = RidgeModel(feature_columns, target_kind, train_on_corrupt_rows=train_on_corrupt_rows)
        self.tree_columns = [column for column in feature_columns if column not in self.TREND_COLUMNS]
        self.target_kind = target_kind
        self.params = {**config.LIGHTGBM_PARAMS, **(params or {})}
        self.n_rounds = n_rounds or config.MAX_BOOSTING_ROUNDS
        self.train_on_corrupt_rows = train_on_corrupt_rows

    def fit(self, train_table):
        """Fit Ridge, then grow n_rounds trees on top of Ridge's predictions."""
        rows = select_training_rows(train_table, self.train_on_corrupt_rows)
        self.ridge.fit(rows)
        starting_point = self.ridge.predict_target_space(rows)
        target = to_model_target(rows["posted_rate"], rows["distance"], self.target_kind)
        dataset = lgb.Dataset(rows[self.tree_columns], target, init_score=starting_point)
        self.booster_ = lgb.train(self.params, dataset, num_boost_round=self.n_rounds)
        return self

    def predict(self, table, n_rounds=None):
        """Predicted price in dollars, using the first n_rounds trees (all if None)."""
        # LightGBM's own prediction excludes the starting point, so add Ridge back.
        correction = self.booster_.predict(table[self.tree_columns], num_iteration=n_rounds)
        raw_prediction = self.ridge.predict_target_space(table) + correction
        return from_model_target(raw_prediction, table["distance"], self.target_kind)

    def feature_importance(self):
        """Gain of each feature in the LightGBM (correction) part, as a Series."""
        gains = self.booster_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.tree_columns).sort_values(ascending=False)


# =============================================================================
# F - ensemble
# =============================================================================
class EnsembleModel:
    """
    Model F: fit several models and average their dollar predictions.
    Different model families make different mistakes, so the average is often
    a little more accurate than any single member.
    """

    def __init__(self, members):
        self.members = members

    def fit(self, train_table):
        """Fit every member (each member decides itself whether to keep corrupt rows)."""
        for member in self.members:
            member.fit(train_table)
        return self

    def predict(self, table):
        """Plain average of the members' predictions, in dollars."""
        return np.mean([member.predict(table) for member in self.members], axis=0)

    def feature_importance(self):
        """Average of the members' importances (each scaled to sum to 1)."""
        shares = []
        for member in self.members:
            if hasattr(member, "feature_importance"):
                importance = member.feature_importance()
                shares.append(importance / importance.sum())
        return pd.concat(shares, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)


# =============================================================================
# Factory
# =============================================================================
def build_model(spec):
    """
    Turn a ModelSpec into an (unfitted) model object.

    Input:  ModelSpec.
    Output: a model with fit(table) and predict(table).
    """
    features = columns_for_groups(spec.feature_groups)
    keep_corrupt = spec.train_on_corrupt_rows

    if spec.model_type == "baseline":
        return BaselineModel(train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "ridge":
        return RidgeModel(features, spec.target_kind, train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "lightgbm":
        return LightGBMModel(features, spec.target_kind, spec.params, objective="regression",
                             n_rounds=spec.n_rounds, train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "lightgbm_huber":
        return LightGBMModel(features, spec.target_kind, spec.params, objective="huber",
                             n_rounds=spec.n_rounds, train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "catboost":
        return CatBoostModel(features, spec.target_kind, spec.params,
                             n_rounds=spec.n_rounds, train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "ridge_lightgbm":
        return RidgeThenLightGBMModel(features, spec.target_kind, spec.params,
                                      n_rounds=spec.n_rounds, train_on_corrupt_rows=keep_corrupt)
    if spec.model_type == "ensemble":
        return EnsembleModel([build_model(member) for member in spec.members])
    raise ValueError(f"unknown model type: {spec.model_type}")
