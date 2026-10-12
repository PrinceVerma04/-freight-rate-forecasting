"""
Phase 2 - feature engineering.

`FeatureBuilder` follows the familiar scikit-learn pattern:

    builder = FeatureBuilder()
    train_table = builder.fit_transform(cleaned_training_rows)  # learns from TRAIN only
    other_table = builder.transform(cleaned_other_rows)         # holdout / validation / December

Every engineered column belongs to a named group in FEATURE_GROUPS, so the
ablation study (Phase 4) can switch a whole idea off with one line.

`prepare_model_inputs()` at the bottom glues Phase 1 + Phase 2 together for one
train/test pair, learning every statistic from the training rows only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src import config
from src.data_cleaning import (
    CleaningStats, clean_features, fit_cleaning_stats, flag_corrupted_targets, haversine_miles,
)

# =============================================================================
# Feature groups
# =============================================================================
FEATURE_GROUPS = {
    # How far the load travels - by far the biggest driver of price.
    # Price grows a bit slower than distance (fixed costs), hence log_distance.
    "distance": ["distance", "log_distance", "haversine_miles", "distance_to_haversine_ratio"],

    # Where the load goes. Built only from coordinates, so it also works for the
    # 8 validation cities that never appear in training (no city names needed).
    "geography": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                  "delta_lat", "delta_lon", "bearing_sin", "bearing_cos",
                  "midpoint_lat", "midpoint_lon", "pickup_region", "delivery_region"],

    # How expensive a city / lane is compared with similar loads (target encoding).
    "city_level": ["pickup_city_level", "delivery_city_level", "lane_level"],

    # Reefer (refrigerated) and flatbed trailers cost ~12% / ~8% more than dry vans.
    "equipment": ["is_reefer", "is_flatbed"],

    # Heavier loads cost more (effect grows from ~15k lbs and flattens above ~40k).
    "weight": ["weight", "weight_was_missing", "weight_was_capped", "weight_x_distance"],

    # The daily market level. Includes a strong weekly cycle (Thursday high,
    # Sunday low) that also explains the day-of-week pattern in prices.
    "market": ["market_index"],

    # Hand-made market interactions. Trees can learn interactions on their own,
    # and products like market x distance mix two effects into one number, which
    # misleads the trees when the market hits levels rarely seen in training.
    "market_interactions": ["market_x_distance", "market_x_reefer", "market_x_flatbed"],

    # Prices climb ~4% through the last month of every quarter and reset
    # when the next quarter starts (found in EDA; NOT explained by market_index).
    "quarter_end": ["is_quarter_end_month", "quarter_end_progress"],

    # The general price level drifted up during 2025 (also not explained by market_index).
    "time_trend": ["months_since_start"],

    "day_of_week": ["day_of_week", "is_weekend"],
    "holidays": ["is_holiday_week", "days_to_nearest_holiday"],

    # A noisy quote indicator provided with the data (tested in the ablation study).
    "quote_signal": ["quote_signal"],
}

ALL_GROUPS = list(FEATURE_GROUPS)

# Groups the ablation study is allowed to switch off. The core groups
# (distance, geography, equipment, weight, market) are always used.
OPTIONAL_GROUPS = ["quote_signal", "city_level", "market_interactions", "quarter_end",
                   "time_trend", "day_of_week", "holidays"]

# Columns that hold category labels rather than numbers.
CATEGORICAL_FEATURES = ["pickup_region", "delivery_region"]

# Non-feature columns carried along in every table (ids, raw values, labels).
META_COLUMNS = ["load_id", "date", "pickup", "delivery", "equipment",
                "posted_rate", "expected_rate", "target_is_corrupt"]


def columns_for_groups(groups):
    """Return the feature column names for a list of group names (in a stable order)."""
    columns = []
    for group in ALL_GROUPS:
        if group in groups:
            columns.extend(FEATURE_GROUPS[group])
    return columns


# =============================================================================
# Target options
# =============================================================================
TARGET_KINDS = ["log", "per_mile", "raw"]


def to_model_target(rate, distance, kind):
    """
    Turn posted_rate (dollars) into what the model actually learns.

        log      -> log(rate): errors become relative (1% off is 1% off for a
                    $300 or a $5,000 load), matching how prices multiply.
        per_mile -> rate / distance: removes most of the distance effect.
        raw      -> rate in dollars, unchanged.
    """
    rate = np.asarray(rate, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if kind == "log":
        return np.log(rate)
    if kind == "per_mile":
        return rate / distance
    if kind == "raw":
        return rate
    raise ValueError(f"unknown target kind: {kind}")


def from_model_target(prediction, distance, kind):
    """Inverse of to_model_target(): turn a model output back into dollars."""
    prediction = np.asarray(prediction, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if kind == "log":
        return np.exp(prediction)
    if kind == "per_mile":
        return prediction * distance
    if kind == "raw":
        return prediction
    raise ValueError(f"unknown target kind: {kind}")


# =============================================================================
# Small geometry helpers
# =============================================================================
def compass_bearing_radians(lat1, lon1, lat2, lon2):
    """Direction of travel from point 1 to point 2 (0 = north, pi/2 = east), in radians."""
    lat1, lon1, lat2, lon2 = (np.radians(value) for value in (lat1, lon1, lat2, lon2))
    delta_lon = lon2 - lon1
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    return np.arctan2(x, y)


def city_coordinates_from_rows(df):
    """
    Build a table city -> (lat, lon) from the pickup and delivery columns of df.

    Each city has exactly one coordinate pair in this dataset (checked in EDA).
    """
    pickups = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    deliveries = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city")


# =============================================================================
# The feature builder
# =============================================================================
class FeatureBuilder:
    """
    Learns feature statistics from training rows and turns any rows into a
    model-ready feature table.

    Learned from TRAINING rows only:
      * KMeans regions of the training cities' coordinates,
      * city and lane price levels (target encodings),
      * the coordinates of every training city (for the unseen-city fallback).
    """

    # ------------------------------------------------------------------ fit
    def fit_transform(self, train_df):
        """
        Learn statistics from cleaned TRAINING rows and return their feature table.

        Input:  DataFrame from Phase 1 (must contain posted_rate, expected_rate
                and target_is_corrupt).
        Output: feature table (META_COLUMNS + every feature column).

        Training rows get "as-of" price levels that only use loads from EARLIER
        dates, so no row ever sees its own label or the future (no leakage).
        """
        self.city_coordinates_ = city_coordinates_from_rows(train_df)

        # Regions: group the training cities into 8 clusters by coordinates.
        # A new city later simply joins the cluster with the nearest centre.
        self.region_model_ = KMeans(n_clusters=config.N_REGIONS, n_init=10,
                                    random_state=config.RANDOM_SEED)
        self.region_model_.fit(self.city_coordinates_[["lat", "lon"]].to_numpy())

        self._fit_price_levels(train_df)
        return self._build_table(train_df, is_training=True)

    def transform(self, df):
        """
        Build the feature table for rows that were NOT used for fitting
        (holdout, validation or December rows).

        Input:  DataFrame cleaned with clean_features().
        Output: feature table (META_COLUMNS that exist + every feature column).
        """
        return self._build_table(df, is_training=False)

    # ----------------------------------------------------- price levels (TE)
    def _fit_price_levels(self, train_df):
        """
        Target encoding: how much more (or less) expensive is each city / lane?

        Raw $/mile mostly reflects trip length (short trips cost more per mile),
        so we first compare each load with the reference price model:
            gap = log(actual price / expected price)
        Then we compare it with the OTHER loads of the SAME DAY:
            relative_gap = gap - average gap of that day
        so market-wide moves (quarter-end ramps, the yearly drift) do not leak
        into the city levels. A city's level is the average relative_gap of its
        loads, shrunk towards 0 when the city has few loads (smoothing).

        EDA showed a city's effect is the same whether it is the pickup or the
        delivery end (correlation 0.99), so every load counts for both of its cities.
        Corrupted labels are ignored here.
        """
        clean = train_df[train_df["target_is_corrupt"] == 0]
        gap = np.log(clean["posted_rate"] / clean["expected_rate"])
        relative_gap = gap - gap.groupby(clean["date"]).transform("mean")

        city_rows = pd.concat([
            pd.DataFrame({"key": clean["pickup"], "date": clean["date"], "gap": relative_gap}),
            pd.DataFrame({"key": clean["delivery"], "date": clean["date"], "gap": relative_gap}),
        ])
        lane_rows = pd.DataFrame({"key": clean["pickup"] + " -> " + clean["delivery"],
                                  "date": clean["date"], "gap": relative_gap})

        # Totals over the whole training period (used for rows we predict).
        self.city_totals_ = city_rows.groupby("key")["gap"].agg(gap_sum="sum", n_loads="count")
        self.lane_totals_ = lane_rows.groupby("key")["gap"].agg(gap_sum="sum", n_loads="count")
        # Day-by-day running totals (used for the leakage-free training encodings).
        self.city_running_totals_ = self._running_totals(city_rows)
        self.lane_running_totals_ = self._running_totals(lane_rows)

        # Price level of every known city, plus a nearest-neighbour estimate helper.
        self.city_level_ = (self.city_totals_["gap_sum"]
                            / (self.city_totals_["n_loads"] + config.CITY_LEVEL_SMOOTHING))

    @staticmethod
    def _running_totals(rows):
        """Cumulative gap sum and load count per key, one row per (key, date)."""
        daily = (rows.groupby(["key", "date"])["gap"].agg(gap_sum="sum", n_loads="count")
                 .reset_index().sort_values("date"))
        daily["cum_sum"] = daily.groupby("key")["gap_sum"].cumsum()
        daily["cum_count"] = daily.groupby("key")["n_loads"].cumsum()
        return daily[["key", "date", "cum_sum", "cum_count"]]

    @staticmethod
    def _as_of_totals(keys, dates, running_totals):
        """
        For every (key, date) pair, look up the running totals of STRICTLY
        EARLIER dates (allow_exact_matches=False), i.e. what was known the day before.
        Returns two numpy arrays: summed gaps and load counts (0 when nothing known).
        """
        query = pd.DataFrame({"key": keys.to_numpy(), "date": dates.to_numpy(),
                              "row_order": np.arange(len(keys))}).sort_values("date")
        matched = pd.merge_asof(query, running_totals, on="date", by="key",
                                allow_exact_matches=False)
        matched = matched.sort_values("row_order")
        return matched["cum_sum"].fillna(0).to_numpy(), matched["cum_count"].fillna(0).to_numpy()

    def nearest_city_level(self, lat, lon):
        """
        Price level for a city never seen in training: the average level of the
        K nearest training cities. EDA showed city levels change smoothly across
        the map (about 84% of their variation is explained by location alone),
        so close neighbours are a sensible stand-in.
        """
        known = self.city_coordinates_.loc[self.city_level_.index]
        distances = haversine_miles(lat, lon, known["lat"].to_numpy(), known["lon"].to_numpy())
        nearest = np.argsort(distances)[:config.N_NEAREST_CITIES]
        return float(self.city_level_.iloc[nearest].mean())

    def _city_levels_for_rows(self, df, is_training):
        """Return (pickup_level, delivery_level, lane_level) arrays for every row."""
        lane_keys = df["pickup"] + " -> " + df["delivery"]
        smoothing_city = config.CITY_LEVEL_SMOOTHING
        smoothing_lane = config.LANE_LEVEL_SMOOTHING

        if is_training:
            # As-of encodings: only loads from earlier dates are used.
            pickup_sum, pickup_count = self._as_of_totals(df["pickup"], df["date"], self.city_running_totals_)
            delivery_sum, delivery_count = self._as_of_totals(df["delivery"], df["date"], self.city_running_totals_)
            lane_sum, lane_count = self._as_of_totals(lane_keys, df["date"], self.lane_running_totals_)
            pickup_level = pickup_sum / (pickup_count + smoothing_city)
            delivery_level = delivery_sum / (delivery_count + smoothing_city)
        else:
            # Rows we predict: use everything learned from training.
            # Unseen cities fall back to the average of their nearest known cities.
            coordinates = city_coordinates_from_rows(df)
            level_by_city = {}
            for city in coordinates.index:
                if city in self.city_level_.index:
                    level_by_city[city] = float(self.city_level_[city])
                else:
                    level_by_city[city] = self.nearest_city_level(coordinates.loc[city, "lat"],
                                                                   coordinates.loc[city, "lon"])
            pickup_level = df["pickup"].map(level_by_city).to_numpy()
            delivery_level = df["delivery"].map(level_by_city).to_numpy()
            lane_sum = lane_keys.map(self.lane_totals_["gap_sum"]).fillna(0).to_numpy()
            lane_count = lane_keys.map(self.lane_totals_["n_loads"]).fillna(0).to_numpy()

        # A lane's level is shrunk towards "pickup level + delivery level", so a
        # lane with few (or zero) loads simply inherits the levels of its cities.
        lane_prior = pickup_level + delivery_level
        lane_level = (lane_sum + smoothing_lane * lane_prior) / (lane_count + smoothing_lane)
        return pickup_level, delivery_level, lane_level

    # --------------------------------------------------------- build table
    def _build_table(self, df, is_training):
        """Compute every feature group for df and return one table."""
        table = df[[column for column in META_COLUMNS if column in df.columns]].copy()

        # ---- distance -------------------------------------------------------
        straight_line = haversine_miles(df["pickup_lat"], df["pickup_lon"],
                                        df["delivery_lat"], df["delivery_lon"])
        table["distance"] = df["distance"]
        table["log_distance"] = np.log(df["distance"])
        table["haversine_miles"] = straight_line
        # Road miles vs straight-line miles. Clipped at 5 because the synthetic
        # coordinates put a few city pairs almost on top of each other.
        table["distance_to_haversine_ratio"] = (df["distance"] / np.maximum(straight_line, 1.0)).clip(upper=5.0)

        # ---- geography (works for unseen cities: coordinates only) ----------
        for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            table[column] = df[column]
        table["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]   # >0 means heading north
        table["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]   # >0 means heading east
        bearing = compass_bearing_radians(df["pickup_lat"], df["pickup_lon"],
                                          df["delivery_lat"], df["delivery_lon"])
        # sin/cos instead of the raw angle, so 359 degrees and 1 degree look alike.
        table["bearing_sin"] = np.sin(bearing)
        table["bearing_cos"] = np.cos(bearing)
        table["midpoint_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
        table["midpoint_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2
        region_labels = list(range(config.N_REGIONS))
        pickup_region = self.region_model_.predict(df[["pickup_lat", "pickup_lon"]].to_numpy())
        delivery_region = self.region_model_.predict(df[["delivery_lat", "delivery_lon"]].to_numpy())
        table["pickup_region"] = pd.Categorical(pickup_region, categories=region_labels)
        table["delivery_region"] = pd.Categorical(delivery_region, categories=region_labels)

        # ---- city / lane price levels (target encoding) ---------------------
        pickup_level, delivery_level, lane_level = self._city_levels_for_rows(df, is_training)
        table["pickup_city_level"] = pickup_level
        table["delivery_city_level"] = delivery_level
        table["lane_level"] = lane_level

        # ---- equipment --------------------------------------------------------
        # One-hot with Dry Van (the most common type) as the baseline.
        table["is_reefer"] = (df["equipment"] == "Reefer").astype(int)
        table["is_flatbed"] = (df["equipment"] == "Flatbed").astype(int)

        # ---- weight -----------------------------------------------------------
        table["weight"] = df["weight"]
        table["weight_was_missing"] = df["weight_was_missing"]
        table["weight_was_capped"] = df["weight_was_capped"]
        # Heavy loads on long trips (scaled to "million lb-miles" for readability).
        table["weight_x_distance"] = df["weight"] * df["distance"] / 1e6

        # ---- market (+ hand-made interactions) --------------------------------
        table["market_index"] = df["market_index"]
        table["market_x_distance"] = df["market_index"] * df["distance"]
        table["market_x_reefer"] = df["market_index"] * table["is_reefer"]
        table["market_x_flatbed"] = df["market_index"] * table["is_flatbed"]

        # ---- calendar ---------------------------------------------------------
        # Why NOT raw month or day_of_year? Training covers Jan-Oct 2025 only, so a
        # month-specific pattern for Nov/Dec can never be learned, and one partial
        # year holds no repeated seasonality. Instead we use features whose values
        # DO occur in training:
        #   * quarter_end_progress: EDA showed prices climb ~4% through the last
        #     month of every quarter (Mar, Jun, Sep) and reset afterwards.
        #     December is also a quarter-end month, so the pattern carries over.
        #   * months_since_start: the overall price level drifted up during 2025.
        #     Tree models cannot extrapolate, so for Nov/Dec they simply re-use the
        #     most recent (October) level - a cautious "persistence" forecast -
        #     instead of inventing a trend.
        # market_index alone does NOT explain either effect (see EDA), which is
        # why these features exist. The ablation study measures their value.
        dates = df["date"]
        is_quarter_end = dates.dt.month.isin([3, 6, 9, 12])
        month_progress = (dates.dt.day - 1) / (dates.dt.days_in_month - 1)
        table["is_quarter_end_month"] = is_quarter_end.astype(int)
        table["quarter_end_progress"] = np.where(is_quarter_end, month_progress, 0.0)
        table["months_since_start"] = ((dates.dt.year - config.DATA_START_DATE.year) * 12
                                       + (dates.dt.month - config.DATA_START_DATE.month))
        table["day_of_week"] = dates.dt.dayofweek          # 0 = Monday ... 6 = Sunday
        table["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

        # Days to the closest freight holiday, capped at one week (beyond that a
        # holiday cannot plausibly matter).
        day_numbers = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
        holiday_numbers = config.US_FREIGHT_HOLIDAYS.to_numpy().astype("datetime64[D]").astype(np.int64)
        days_to_holiday = np.abs(day_numbers[:, None] - holiday_numbers[None, :]).min(axis=1)
        table["days_to_nearest_holiday"] = np.minimum(days_to_holiday, config.HOLIDAY_DISTANCE_CAP_DAYS)
        table["is_holiday_week"] = (days_to_holiday <= config.HOLIDAY_WINDOW_DAYS).astype(int)

        # ---- quote signal -----------------------------------------------------
        table["quote_signal"] = df["quote_signal"]

        return table


# =============================================================================
# Phase 1 + Phase 2 for one train/test pair
# =============================================================================
@dataclass
class ModelInputs:
    """Model-ready tables plus the fitted objects needed to process new rows."""
    train_table: pd.DataFrame
    test_table: pd.DataFrame
    cleaning_stats: CleaningStats
    feature_builder: FeatureBuilder


def prepare_model_inputs(train_rows, test_rows=None, verbose=False):
    """
    Clean + engineer features for one train/test pair WITHOUT leakage.

    Every statistic (medians, lane distances, reference prices, regions, city
    levels) is learned from train_rows only and then applied to test_rows.

    Input:  raw training rows (with posted_rate) and optional raw test rows.
    Output: ModelInputs. If the test rows carry posted_rate (a holdout), their
            labels are flagged as corrupt or not - for EVALUATION only.
    """
    stats = fit_cleaning_stats(train_rows)

    train_clean = clean_features(train_rows, stats, verbose)
    train_clean = flag_corrupted_targets(train_clean, stats.reference_model, verbose)
    builder = FeatureBuilder()
    train_table = builder.fit_transform(train_clean)

    test_table = None
    if test_rows is not None:
        test_clean = clean_features(test_rows, stats, verbose)
        if "posted_rate" in test_clean.columns:
            test_clean = flag_corrupted_targets(test_clean, stats.reference_model, verbose=False)
        test_table = builder.transform(test_clean)

    return ModelInputs(train_table, test_table, stats, builder)


def prepare_new_rows(rows, cleaning_stats, feature_builder, verbose=False):
    """
    Clean + engineer features for brand-new rows (e.g. the December chart
    inputs) using objects that were already fitted on training data.
    """
    cleaned = clean_features(rows, cleaning_stats, verbose)
    return feature_builder.transform(cleaned)
