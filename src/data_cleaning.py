"""
Phase 1 - data cleaning.

One function per data-quality issue. Every function:
  * receives a DataFrame and returns a NEW DataFrame (the input is never
    changed in place), and
  * prints how many rows it touched (when verbose=True), so the run log
    doubles as an audit trail for the report.

Leakage rule
------------
Anything *learned* from data (median weights, typical lane distances, the
reference price model) is learned by `fit_cleaning_stats()` from TRAINING
rows only and then re-used, unchanged, on holdout / validation / December rows.

The one deliberate exception is `fill_missing_market_index()`: market_index is
an input feature that is known for every load on its own date, so a missing
value is filled from other loads of the SAME DATE in the SAME FILE. That uses
no labels (no posted_rate), so it cannot leak the answer.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


# =============================================================================
# Small helpers
# =============================================================================
def log_step(message, verbose):
    """Print one indented log line, but only when verbose is True."""
    if verbose:
        print(f"    - {message}")


def load_raw_data():
    """
    Read the two provided CSV files.

    Returns:
        (train, validation): DataFrames with the 'date' column parsed as datetime.
    """
    train = pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])
    validation = pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"])
    return train, validation


def haversine_miles(lat1, lon1, lat2, lon2):
    """
    Straight-line ("as the crow flies") distance in miles between two points.

    Inputs can be single numbers or numpy arrays / pandas Series of degrees.
    Returns the distance(s) in miles.
    """
    earth_radius_miles = 3958.8
    lat1, lon1, lat2, lon2 = (np.radians(value) for value in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * earth_radius_miles * np.arcsin(np.sqrt(a))


# =============================================================================
# Issue 1 - weight: negative values, a hard cap, missing values
# =============================================================================
def fix_negative_weights(df, verbose=True):
    """
    Make negative weights positive.

    Why abs()? The negative values are sign-flip errors: their magnitudes have
    the same distribution as normal weights (median ~31.8k lbs vs ~31.4k lbs),
    and a load cannot weigh less than nothing.

    Input:  DataFrame with a 'weight' column.
    Output: copy with fixed weights + flag column 'weight_was_negative'.
    """
    out = df.copy()
    is_negative = out["weight"] < 0
    out["weight_was_negative"] = is_negative.astype(int)
    out.loc[is_negative, "weight"] = out.loc[is_negative, "weight"].abs()
    log_step(f"weight: {is_negative.sum():,} negative values made positive (sign-flip errors)", verbose)
    return out


def flag_capped_weights(df, verbose=True):
    """
    Flag loads whose weight sits exactly on the 47,500 lb cap.

    The true weight of these loads is "47,500 or more" - it cannot be recovered,
    but the flag tells the model the value is censored. (The price effect of
    weight flattens out above ~40k lbs anyway, so little information is lost.)
    Must run AFTER fix_negative_weights: 13 capped values were also sign-flipped.

    Input:  DataFrame with a 'weight' column.
    Output: copy with the flag column 'weight_was_capped'.
    """
    out = df.copy()
    out["weight_was_capped"] = (out["weight"] == config.WEIGHT_CAP_LBS).astype(int)
    log_step(f"weight: {out['weight_was_capped'].sum():,} loads flagged as capped at "
             f"{config.WEIGHT_CAP_LBS:,} lbs", verbose)
    return out


def fit_weight_medians(train_df):
    """
    Learn the median weight of each equipment type from TRAINING rows.

    Input:  training DataFrame.
    Output: dict such as {"Dry Van": 31444.0, "Flatbed": 31532.5, "Reefer": 31577.0}.
    """
    # abs() first, so the sign-flip errors do not drag the medians down.
    weights = train_df["weight"].abs()
    return weights.groupby(train_df["equipment"]).median().to_dict()


def impute_missing_weights(df, median_weight_by_equipment, verbose=True):
    """
    Fill missing weights with the TRAINING median for the same equipment type.

    Input:  DataFrame + the dict returned by fit_weight_medians().
    Output: copy with no missing weights + flag column 'weight_was_missing'
            (so the model can still tell that the value was a guess).
    """
    out = df.copy()
    is_missing = out["weight"].isna()
    out["weight_was_missing"] = is_missing.astype(int)

    fill_values = out["equipment"].map(median_weight_by_equipment)
    # Fallback for an equipment type never seen in training: the overall median.
    overall_median = float(np.median(list(median_weight_by_equipment.values())))
    fill_values = fill_values.fillna(overall_median)

    out.loc[is_missing, "weight"] = fill_values[is_missing]
    log_step(f"weight: {is_missing.sum():,} missing values filled with the training median "
             f"of their equipment type", verbose)
    return out


# =============================================================================
# Issue 2 - distance: check for corrupted (wildly wrong) values
# =============================================================================
def fit_lane_distances(train_df):
    """
    Learn what a "normal" distance looks like, from TRAINING rows.

    Input:  training DataFrame.
    Output: (lane_median_distance, road_factor)
        lane_median_distance - Series indexed by (pickup, delivery) -> median miles
        road_factor          - typical road-miles / straight-line-miles ratio, used
                               for lanes that never appear in training
    """
    lane_median_distance = train_df.groupby(["pickup", "delivery"])["distance"].median()

    straight_line = haversine_miles(train_df["pickup_lat"], train_df["pickup_lon"],
                                    train_df["delivery_lat"], train_df["delivery_lon"])
    # Only lanes clearly above the 70-mile floor give a meaningful ratio.
    long_enough = straight_line > config.MIN_DISTANCE_MILES
    road_factor = float(np.median(train_df.loc[long_enough, "distance"] / straight_line[long_enough]))
    return lane_median_distance, road_factor


def fix_extreme_distances(df, lane_median_distance, road_factor, verbose=True):
    """
    Detect and repair distances that are wildly inconsistent with their lane.

    Expected distance of a load =
        * the TRAINING median distance of the same lane, if the lane is known;
        * otherwise max(70, straight-line distance x road_factor).
    A distance more than 3x above (or below) that expectation is replaced by the
    expectation and flagged in 'distance_was_fixed'.

    Why not simply compare with the straight-line distance? Because the city
    coordinates in this dataset are synthetic: New Orleans and Shreveport sit
    only 7 miles apart, so their (correct, floored) 70-mile loads would look
    "9.6x too long". Comparing with the lane's own typical distance avoids
    those false alarms.

    Input:  DataFrame + outputs of fit_lane_distances().
    Output: copy with repaired distances + flag column 'distance_was_fixed'.
    """
    out = df.copy()
    straight_line = haversine_miles(out["pickup_lat"], out["pickup_lon"],
                                    out["delivery_lat"], out["delivery_lon"])
    naive_flag_count = int((out["distance"] > config.DISTANCE_RATIO_LIMIT * straight_line).sum())

    # Expected distance: the lane's training median ...
    lane_keys = pd.MultiIndex.from_frame(out[["pickup", "delivery"]])
    expected = pd.Series(lane_median_distance.reindex(lane_keys).to_numpy(), index=out.index)
    # ... or, for a lane never seen in training, a road-distance estimate.
    unseen_lane_estimate = np.maximum(config.MIN_DISTANCE_MILES, straight_line * road_factor)
    expected = expected.fillna(unseen_lane_estimate)

    ratio = out["distance"] / expected
    is_extreme = (ratio > config.DISTANCE_RATIO_LIMIT) | (ratio < 1 / config.DISTANCE_RATIO_LIMIT)
    out["distance_was_fixed"] = is_extreme.astype(int)
    out.loc[is_extreme, "distance"] = expected[is_extreme].round(1)

    log_step(f"distance: {is_extreme.sum():,} extreme values replaced (ratio to lane median "
             f"ranges {ratio.min():.2f}-{ratio.max():.2f}); a naive 'distance > 3x straight-line' "
             f"check would have wrongly flagged {naive_flag_count:,} legitimate "
             f"{config.MIN_DISTANCE_MILES}-mile-floor loads", verbose)
    return out


# =============================================================================
# Issue 3 - market_index: missing values
# =============================================================================
def fill_missing_market_index(df, verbose=True):
    """
    Fill missing market_index values with the median of the SAME DATE.

    market_index is essentially a daily market reading (loads on the same day
    differ by only ~0.025, while the daily level moves between ~0.8 and ~1.3),
    so other loads from the same day are the best guess. If a whole date had no
    reading at all, the value is interpolated from the neighbouring dates.

    Uses only this file's own feature values (no labels), so it is leakage-free.

    Input:  DataFrame with 'date' and 'market_index'.
    Output: copy with no missing market_index + flag 'market_index_was_missing'.
    """
    out = df.copy()
    is_missing = out["market_index"].isna()
    out["market_index_was_missing"] = is_missing.astype(int)

    daily_median = out.groupby("date")["market_index"].median()
    # Make sure every calendar day exists, then interpolate days with no reading.
    all_days = pd.date_range(daily_median.index.min(), daily_median.index.max(), freq="D")
    daily_median = daily_median.reindex(all_days)
    n_days_without_reading = int(daily_median.isna().sum())
    daily_median = daily_median.interpolate(method="time").ffill().bfill()

    out.loc[is_missing, "market_index"] = out.loc[is_missing, "date"].map(daily_median)
    log_step(f"market_index: {is_missing.sum():,} missing values filled with the same-day median "
             f"({n_days_without_reading} days needed interpolation)", verbose)
    return out


# =============================================================================
# Issue 4 - posted_rate: corrupted labels (training data only)
# =============================================================================
class ReferencePriceModel:
    """
    A deliberately simple "expected price" for a load:

        expected_rate = distance
                        x typical_rate_per_mile(equipment, distance)
                        x (market_index / typical_market_index) ** market_elasticity

    * typical_rate_per_mile - the MEDIAN $/mile inside distance bands, per
      equipment type, linearly interpolated between the band centres. Medians
      are used on purpose: they are not pulled around by the corrupted labels
      we are trying to find.
    * market_elasticity - how strongly prices follow market_index, learned with
      a straight-line fit on the log scale after dropping obviously broken rows.
      (It is only ~0.14: a 10% higher market index means ~1.4% higher prices.)

    It is used three times: to spot corrupted labels, as "Model A" (the
    baseline), and as the reference for the city price-level features.
    """

    def fit(self, df):
        """
        Learn the rate-per-mile curves and the market elasticity.

        Input:  DataFrame with distance, equipment, market_index, posted_rate.
        Output: self (fitted).
        """
        rate_per_mile = df["posted_rate"] / df["distance"]
        distance_band = pd.cut(df["distance"], bins=config.DISTANCE_BAND_EDGES, include_lowest=True)
        table = df.assign(rate_per_mile=rate_per_mile, distance_band=distance_band)

        # One curve per equipment type: (log of band-centre distance, median $/mile).
        self.curves_ = {}
        for equipment, rows in table.groupby("equipment"):
            bands = rows.groupby("distance_band", observed=True).agg(
                band_centre=("distance", "median"), median_rate_per_mile=("rate_per_mile", "median"))
            self.curves_[equipment] = (np.log(bands["band_centre"].to_numpy()),
                                       bands["median_rate_per_mile"].to_numpy())

        # Market elasticity = slope of log(actual / curve) against log(market).
        self.typical_market_ = float(df["market_index"].median())
        log_gap = np.log(rate_per_mile.to_numpy() / self._curve_rate_per_mile(df))
        log_market = np.log(df["market_index"].to_numpy() / self.typical_market_)
        usable = np.abs(log_gap) < 0.5  # ignore obviously corrupted labels in this fit
        slope = np.polyfit(log_market[usable], log_gap[usable], deg=1)[0]
        self.market_elasticity_ = float(slope)
        return self

    def _curve_rate_per_mile(self, df):
        """Typical $/mile for each row's equipment type and distance (no market effect)."""
        log_distance = np.log(df["distance"].to_numpy())
        equipment = df["equipment"].to_numpy()
        result = np.full(len(df), np.nan)
        for equipment_type, (x_points, y_points) in self.curves_.items():
            is_this_type = equipment == equipment_type
            result[is_this_type] = np.interp(log_distance[is_this_type], x_points, y_points)
        # An equipment type never seen in training gets the average of all curves.
        if np.isnan(result).any():
            average_curve = np.mean([np.interp(log_distance, x, y) for x, y in self.curves_.values()], axis=0)
            result = np.where(np.isnan(result), average_curve, result)
        return result

    def predict(self, df):
        """
        Expected price in dollars for every row.

        Input:  DataFrame with distance, equipment, market_index.
        Output: numpy array of expected posted_rate values.
        """
        market_adjustment = (df["market_index"].to_numpy() / self.typical_market_) ** self.market_elasticity_
        return df["distance"].to_numpy() * self._curve_rate_per_mile(df) * market_adjustment


def flag_corrupted_targets(df, reference_model, verbose=True):
    """
    Flag posted_rate values that look corrupted.

    ratio = actual posted_rate / expected price from the reference model.
    Clean loads have ratios of roughly 0.84-1.21, while the corrupted ones were
    multiplied by ~0.15-0.5 or ~2-6. Anything below 0.6 or above 1.8 is flagged
    (both thresholds sit in the empty gap between the clusters).

    Rows are only FLAGGED here; the model decides whether to drop them (most
    models) or keep them (the Huber-loss experiment). On holdout rows the flag
    is used for evaluation only ("score on clean labels").

    Input:  DataFrame with posted_rate + a fitted ReferencePriceModel.
    Output: copy with columns expected_rate, target_ratio, target_is_corrupt.
    """
    out = df.copy()
    out["expected_rate"] = reference_model.predict(out)
    out["target_ratio"] = out["posted_rate"] / out["expected_rate"]

    too_low = out["target_ratio"] < config.CORRUPT_RATIO_LOW
    too_high = out["target_ratio"] > config.CORRUPT_RATIO_HIGH
    out["target_is_corrupt"] = (too_low | too_high).astype(int)

    in_grey_zone = (out["target_ratio"].between(0.5, 0.8) | out["target_ratio"].between(1.25, 2.0))
    n_flagged = int(out["target_is_corrupt"].sum())
    log_step(f"posted_rate: {too_low.sum():,} too low (< {config.CORRUPT_RATIO_LOW}x expected) + "
             f"{too_high.sum():,} too high (> {config.CORRUPT_RATIO_HIGH}x) = {n_flagged:,} "
             f"({100 * n_flagged / len(out):.2f}%) flagged as corrupted", verbose)
    log_step(f"posted_rate: {in_grey_zone.sum():,} loads in the ambiguous zone (0.5-0.8x or 1.25-2.0x), "
             f"so the clean and corrupted clusters are cleanly separated", verbose)
    return out


# =============================================================================
# Putting it together
# =============================================================================
@dataclass
class CleaningStats:
    """Everything the cleaning step learns from training rows."""
    median_weight_by_equipment: dict
    lane_median_distance: pd.Series
    road_factor: float
    reference_model: ReferencePriceModel


def clean_features(df, stats, verbose=True):
    """
    Apply every FEATURE fix (weight, distance, market_index) to a set of rows.

    It never looks at posted_rate, so it is safe for holdout, validation and
    December rows. All learned values come from `stats` (fitted on training rows).

    Input:  raw DataFrame + CleaningStats.
    Output: cleaned copy with the flag columns added.
    """
    out = fix_negative_weights(df, verbose)
    out = flag_capped_weights(out, verbose)
    out = impute_missing_weights(out, stats.median_weight_by_equipment, verbose)
    out = fix_extreme_distances(out, stats.lane_median_distance, stats.road_factor, verbose)
    out = fill_missing_market_index(out, verbose)
    return out


def fit_cleaning_stats(train_df):
    """
    Learn all cleaning statistics from TRAINING rows only.

    Input:  raw training DataFrame (must include posted_rate).
    Output: CleaningStats.
    """
    weight_medians = fit_weight_medians(train_df)
    lane_median_distance, road_factor = fit_lane_distances(train_df)

    # The reference price model needs clean distances and a complete market_index,
    # so clean the training rows first (silently) with the statistics just learned.
    partial_stats = CleaningStats(weight_medians, lane_median_distance, road_factor, reference_model=None)
    cleaned_train = clean_features(train_df, partial_stats, verbose=False)
    reference_model = ReferencePriceModel().fit(cleaned_train)

    return CleaningStats(weight_medians, lane_median_distance, road_factor, reference_model)
