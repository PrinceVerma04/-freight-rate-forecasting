"""
Phase 6 - predictions for the fixed December chart.

december-chart-inputs.csv describes ONE load - Lexington -> Fort Wayne, 360
miles, Dry Van, 32,000 lb - on every day of December 2025. Only the date
changes. The file lacks some inputs the model needs, so they are filled
explicitly (and nothing else is changed):

  * pickup / delivery lat & lon -> looked up from training (both cities appear there)
  * market_index               -> the daily median of validation.csv loads on that
                                  same date (validation.csv covers all of December);
                                  interpolated if a date were missing
  * quote_signal               -> the median of this lane in training (only matters
                                  if the final model uses quote_signal)
  * load_id                    -> a made-up id (DEC-2025-12-01, ...), never written out

The output file keeps the original 7 columns, values and order; only
predicted_rate is filled in.
"""

import numpy as np
import pandas as pd

from src import config, plotting
from src.features import city_coordinates_from_rows, prepare_new_rows


def daily_market_index_for(dates, validation_raw):
    """
    market_index for each requested date = median over validation.csv loads of that date.
    Missing dates (none in practice) are interpolated in time from their neighbours.
    """
    daily_median = validation_raw.groupby("date")["market_index"].median()
    all_days = pd.date_range(min(daily_median.index.min(), dates.min()),
                             max(daily_median.index.max(), dates.max()), freq="D")
    daily_median = daily_median.reindex(all_days).interpolate(method="time").ffill().bfill()
    return dates.map(daily_median).to_numpy()


def build_december_model_rows(december_raw, train_raw, validation_raw):
    """
    Turn the 31 December rows into rows the model can read.

    Input:  the raw December file + raw training and validation rows.
    Output: DataFrame with every raw column the cleaning / feature steps expect.
    """
    rows = december_raw.drop(columns=["predicted_rate"]).copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows["distance"] = rows["distance"].astype(float)
    rows["weight"] = rows["weight"].astype(float)
    rows["load_id"] = "DEC-" + rows["date"].dt.strftime("%Y-%m-%d")

    # Coordinates: both cities exist in the training data.
    coordinates = city_coordinates_from_rows(train_raw)
    for end in ["pickup", "delivery"]:
        missing = set(rows[end]) - set(coordinates.index)
        if missing:
            raise ValueError(f"no coordinates in training for {missing}")
        rows[f"{end}_lat"] = rows[end].map(coordinates["lat"])
        rows[f"{end}_lon"] = rows[end].map(coordinates["lon"])

    # market_index: what the market looked like on that day, from validation.csv.
    rows["market_index"] = daily_market_index_for(rows["date"], validation_raw)

    # quote_signal: typical value for this lane (falls back to the overall median).
    same_lane = train_raw[(train_raw["pickup"] == rows["pickup"].iloc[0])
                          & (train_raw["delivery"] == rows["delivery"].iloc[0])]
    lane_median = same_lane["quote_signal"].median() if len(same_lane) >= 5 else train_raw["quote_signal"].median()
    rows["quote_signal"] = float(lane_median)
    return rows


def check_december_output(output, december_raw):
    """The checks score.py runs, done before saving. Raises ValueError on a problem."""
    problems = []
    if list(output.columns) != list(december_raw.columns):
        problems.append("columns changed")
    if len(output) != 31 or output["date"].duplicated().any():
        problems.append("must have exactly one row per December day")
    unchanged = output.drop(columns=["predicted_rate"]).equals(december_raw.drop(columns=["predicted_rate"]))
    if not unchanged:
        problems.append("input values changed")
    if output["predicted_rate"].isna().any() or (output["predicted_rate"] <= 0).any():
        problems.append("predictions must be positive numbers")
    if problems:
        raise ValueError("December output failed its checks: " + "; ".join(problems))
    print("    - checks passed: 31 days, original 7 columns and values unchanged, all predictions > 0")


def predict_december(bundle, train_raw, validation_raw):
    """
    Predict the 31 December rows, save the filled file and a diagnostic figure.

    Output: the filled December DataFrame (original columns + predicted_rate).
    """
    december_raw = pd.read_csv(config.DECEMBER_INPUTS_PATH)
    model_rows = build_december_model_rows(december_raw, train_raw, validation_raw)
    table = prepare_new_rows(model_rows, bundle["cleaning_stats"], bundle["feature_builder"])
    predictions = bundle["model"].predict(table)

    output = december_raw.copy()
    output["predicted_rate"] = np.round(predictions, 2)
    check_december_output(output, december_raw)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.DECEMBER_PREDICTIONS_PATH, index=False)
    print(f"    - saved {config.DECEMBER_PREDICTIONS_PATH.relative_to(config.PROJECT_ROOT)} "
          f"(original december-chart-inputs.csv left untouched)")

    print(f"    - inputs filled: lat/lon from training; market_index = validation daily median "
          f"({model_rows['market_index'].min():.3f}-{model_rows['market_index'].max():.3f}); "
          f"quote_signal = {model_rows['quote_signal'].iloc[0]:.3f}")
    first_week = predictions[:7].mean()
    last_week = predictions[-7:].mean()
    print(f"    - predicted rate: ${predictions.min():,.0f} to ${predictions.max():,.0f} "
          f"(mean ${predictions.mean():,.0f}); first-week average ${first_week:,.0f} -> last-week average "
          f"${last_week:,.0f} ({100 * (last_week / first_week - 1):+.1f}%)")
    weekday_means = pd.Series(predictions, index=model_rows["date"].dt.day_name()).groupby(level=0).mean()
    print(f"    - by weekday: highest {weekday_means.idxmax()} (${weekday_means.max():,.0f}), lowest "
          f"{weekday_means.idxmin()} (${weekday_means.min():,.0f}) - follows the market_index weekly cycle")

    plotting.plot_december_predictions(model_rows["date"].to_numpy(), predictions,
                                       model_rows["market_index"].to_numpy())
    return output, model_rows, predictions
