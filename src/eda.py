"""
Phase 0 - exploratory data analysis (EDA).

Prints the key facts about the raw data and saves the figures used in the
report and the Loom walkthrough. Nothing here changes the data on disk.

The headline findings (all verified below):
  * the data has a handful of simple corruptions (sign-flipped / capped /
    missing weights, missing market_index, ~1.4% corrupted prices);
  * the "too long" distances are NOT corrupted - they are 70-mile-floor loads
    between cities whose synthetic coordinates are almost identical;
  * prices follow market_index only weakly (elasticity ~0.14) and have a
    calendar pattern market_index does not explain: a ~4% ramp through every
    quarter-end month plus an upward drift over the year;
  * quote_signal changes "regime" from month to month, and Nov/Dec look like
    August, the one month where it has no relationship with price;
  * 8 validation cities never appear in training, but city price levels vary
    smoothly with location, so neighbours give a good estimate.
"""

import numpy as np
import pandas as pd

from src import config, plotting
from src.data_cleaning import clean_features, fit_cleaning_stats, flag_corrupted_targets, haversine_miles
from src.features import FeatureBuilder, city_coordinates_from_rows


def describe_raw_data(train, validation):
    """
    Print the basic facts about both files and return them as a dict.

    Input:  raw training and validation DataFrames.
    Output: dict of key numbers (used in the run summary / report).
    """
    facts = {}
    print(f"    - train-test.csv : {len(train):,} rows, {train['date'].min():%Y-%m-%d} to "
          f"{train['date'].max():%Y-%m-%d}")
    print(f"    - validation.csv : {len(validation):,} rows, {validation['date'].min():%Y-%m-%d} to "
          f"{validation['date'].max():%Y-%m-%d}")

    # Cities and coordinates
    train_cities = set(train["pickup"]) | set(train["delivery"])
    validation_cities = set(validation["pickup"]) | set(validation["delivery"])
    unseen_cities = sorted(validation_cities - train_cities)
    both = pd.concat([train, validation])
    coordinates = pd.concat([
        both[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        both[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)])
    max_coordinates_per_city = int(coordinates.drop_duplicates().groupby("city").size().max())
    facts["unseen_cities"] = unseen_cities
    print(f"    - cities: {len(train_cities)} in training, {len(validation_cities)} in validation, "
          f"{len(unseen_cities)} only in validation: {', '.join(unseen_cities)}")
    print(f"    - every city has exactly {max_coordinates_per_city} coordinate pair (coordinates are "
          f"synthetic but consistent)")

    # Weight
    for name, df in [("train", train), ("validation", validation)]:
        weight = df["weight"]
        n_negative = int((weight < 0).sum())
        n_capped = int((weight.abs() == config.WEIGHT_CAP_LBS).sum())
        n_floor = int((weight.abs() == config.WEIGHT_FLOOR_LBS).sum())
        n_missing = int(weight.isna().sum())
        facts[f"{name}_weight"] = {"negative": n_negative, "capped": n_capped, "floor": n_floor, "missing": n_missing}
        print(f"    - weight ({name}): {n_missing} missing, {n_negative} negative, {n_capped:,} at the "
              f"47,500 cap, {n_floor} at the 5,000 floor")

    # Distance
    straight_line = haversine_miles(train["pickup_lat"], train["pickup_lon"],
                                    train["delivery_lat"], train["delivery_lon"])
    lane_median = train.groupby(["pickup", "delivery"])["distance"].transform("median")
    ratio_to_lane = train["distance"] / lane_median
    n_floor = int((train["distance"] == 70).sum())
    looks_too_long = train["distance"] > 3 * straight_line
    n_naive = int(looks_too_long.sum())
    n_naive_at_floor = int((looks_too_long & (train["distance"] == 70)).sum())
    print(f"    - distance: min {train['distance'].min():.0f} miles ({n_floor} loads at the 70-mile floor); "
          f"{n_naive} loads are >3x the straight-line distance, and {n_naive_at_floor} of them are "
          f"70-mile-floor loads between cities with almost identical synthetic coordinates; distance vs "
          f"its lane median ranges {ratio_to_lane.min():.2f}-{ratio_to_lane.max():.2f}x "
          f"-> no corrupted distances")

    # Market index
    within_day = both.groupby("date")["market_index"].std().mean()
    between_days = both.groupby("date")["market_index"].median().std()
    facts["market_mean_train"] = float(train["market_index"].mean())
    facts["market_mean_validation"] = float(validation["market_index"].mean())
    print(f"    - market_index: {train['market_index'].isna().sum()} missing in train, "
          f"{validation['market_index'].isna().sum()} in validation; mean {facts['market_mean_train']:.3f} "
          f"(train) vs {facts['market_mean_validation']:.3f} (validation); spread within a day "
          f"{within_day:.3f} vs between days {between_days:.3f} -> essentially a daily value")
    return facts


def build_eda_figures(train, validation, facts):
    """
    Create the EDA figures and print the numbers behind them.

    Input:  raw training / validation rows + the facts dict (updated in place).
    Output: list of saved figure paths.
    """
    paths = []
    stats = fit_cleaning_stats(train)
    train_clean = flag_corrupted_targets(clean_features(train, stats, verbose=False),
                                         stats.reference_model, verbose=False)
    validation_clean = clean_features(validation, stats, verbose=False)
    facts["market_elasticity"] = stats.reference_model.market_elasticity_
    print(f"    - price vs market: elasticity {facts['market_elasticity']:.3f} "
          f"(10% higher market_index -> ~{10 * facts['market_elasticity']:.1f}% higher price)")

    # 1) Corrupted labels -------------------------------------------------------
    paths.append(plotting.plot_target_ratio_histogram(train_clean["target_ratio"].to_numpy(),
                                                      low=config.CORRUPT_RATIO_LOW, high=config.CORRUPT_RATIO_HIGH))

    # 2) Daily price level vs the reference price ----------------------------------
    clean = train_clean[train_clean["target_is_corrupt"] == 0]
    log_gap = np.log(clean["posted_rate"] / clean["expected_rate"])
    daily_level = 100 * log_gap.groupby(clean["date"]).median()
    daily_level = daily_level - daily_level.mean()  # centre on 0 = "average day"
    daily_market = pd.concat([train_clean, validation_clean]).groupby("date")["market_index"].median()
    paths.append(plotting.plot_daily_price_level(daily_level, daily_market))

    ramps = []
    for month in (3, 6, 9):
        in_month = daily_level[daily_level.index.month == month]
        ramps.append(in_month.tail(5).mean() - in_month.head(5).mean())
    level_jan_feb = daily_level[daily_level.index.month <= 2].mean()
    level_oct = daily_level[daily_level.index.month == 10].mean()
    facts["quarter_end_ramp_pct"] = float(np.mean(ramps))
    facts["level_drift_pct"] = float(level_oct - level_jan_feb)
    print(f"    - calendar: prices rise {np.mean(ramps):.1f}% from the start to the end of each "
          f"quarter-end month (Mar {ramps[0]:.1f}%, Jun {ramps[1]:.1f}%, Sep {ramps[2]:.1f}%); "
          f"October is {level_oct - level_jan_feb:.1f}% above Jan-Feb even after market_index")

    # 3) quote_signal regimes ------------------------------------------------------
    both = pd.concat([train, validation])
    months = both["date"].dt.to_period("M")
    monthly_values = [(period.strftime("%b"), both.loc[months == period, "quote_signal"].to_numpy(),
                       period.month >= 11) for period in sorted(months.unique())]
    paths.append(plotting.plot_quote_signal_by_month(monthly_values))
    august = clean[clean["date"].dt.month == 8]
    august_corr = np.corrcoef(august["quote_signal"], np.log(august["posted_rate"] / august["expected_rate"]))[0, 1]
    print(f"    - quote_signal: 1%/50%/99% quantiles Aug "
          f"{np.percentile(both.loc[months.dt.month == 8, 'quote_signal'], [1, 50, 99]).round(2)}, "
          f"Nov-Dec {np.percentile(validation['quote_signal'], [1, 50, 99]).round(2)}; "
          f"correlation with price (vs reference) in August = {august_corr:.3f}")

    # 4) City price levels (incl. the 8 unseen cities) ------------------------------
    builder = FeatureBuilder()
    builder.fit_transform(train_clean)
    city_coordinates = city_coordinates_from_rows(pd.concat([train, validation]))
    rows = []
    for city, (lat, lon) in city_coordinates[["lat", "lon"]].iterrows():
        is_unseen = city in facts["unseen_cities"]
        level = builder.nearest_city_level(lat, lon) if is_unseen else builder.city_level_[city]
        rows.append({"city": city, "lat": lat, "lon": lon, "level_pct": 100 * level, "is_unseen": is_unseen})
    city_table = pd.DataFrame(rows).set_index("city")
    paths.append(plotting.plot_city_price_levels(city_table))
    seen = city_table[~city_table["is_unseen"]]
    print(f"    - city price levels range {seen['level_pct'].min():+.1f}% to {seen['level_pct'].max():+.1f}% "
          f"(cheapest {seen['level_pct'].idxmin()}, priciest {seen['level_pct'].idxmax()})")
    return paths


def run_eda(train, validation):
    """
    Run the whole EDA phase.

    Output: dict of key facts.
    """
    facts = describe_raw_data(train, validation)
    paths = build_eda_figures(train, validation, facts)
    print(f"    - saved {len(paths)} figures to reports/figures/")
    return facts
