"""
Plot helpers: one consistent, quiet chart style for every figure in the report.

Design rules used throughout:
  * colours come from a colour-blind-validated palette (only its first three
    categorical colours are used, which stay distinguishable for every type of
    colour blindness),
  * thin lines, solid hairline gridlines, no chart junk,
  * never two y-axes on one plot - two measures get two stacked panels.

Every function saves a PNG into reports/figures/ and returns its path.
"""

import matplotlib

matplotlib.use("Agg")  # draw straight to files, never open a window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src import config

# ---- palette ------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SHADE = "#efeee9"          # background band (e.g. quarter-end months)
BLUE = "#2a78d6"           # categorical slot 1
ORANGE = "#eb6834"         # categorical slot 2
AQUA = "#1baf7a"           # categorical slot 3
DIVERGING_LOW = "#2a78d6"  # cheaper than average
DIVERGING_MID = "#f0efec"  # neutral grey midpoint
DIVERGING_HIGH = "#e34948" # pricier than average


def apply_style():
    """Set matplotlib defaults once, so every figure looks the same."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS_LINE,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDLINE, "grid.linewidth": 0.7,
        "grid.linestyle": "-", "xtick.color": AXIS_LINE, "ytick.color": AXIS_LINE,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "text.color": INK, "legend.frameon": False, "lines.linewidth": 2.0,
    })


def save_figure(figure, filename):
    """Save a figure into reports/figures/ and close it. Returns the file path."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def note(axis, text, y=-0.18):
    """Small grey footnote under a chart (data source / reading guide)."""
    axis.text(0, y, text, transform=axis.transAxes, fontsize=8.5, color=INK_MUTED, va="top")


def format_month_axis(axis):
    """Month ticks like 'Jan', 'Feb', ... on a date axis."""
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def shade_quarter_end_months(axis, start, end):
    """Grey bands behind March, June, September, December."""
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            band_start = np.datetime64(f"{year}-{month:02d}-01")
            band_end = band_start.astype("datetime64[M]") + np.timedelta64(1, "M")
            if band_start <= np.datetime64(end) and band_end.astype("datetime64[D]") >= np.datetime64(start):
                axis.axvspan(band_start, band_end.astype("datetime64[D]"), color=SHADE, zorder=0, lw=0)


# =============================================================================
# Phase 0 / 1 figures
# =============================================================================
def plot_target_ratio_histogram(target_ratio, low, high):
    """
    Histogram of actual / expected price for every training load, on log axes,
    showing the clean cluster around 1.0 and the two corrupted clusters.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=(9.5, 4.2))
    bins = np.geomspace(0.1, 8, 140)
    is_flagged = (target_ratio < low) | (target_ratio > high)
    axis.hist(target_ratio[~is_flagged], bins=bins, color=BLUE, label=f"kept ({(~is_flagged).sum():,} loads)")
    axis.hist(target_ratio[is_flagged], bins=bins, color=ORANGE,
              label=f"flagged as corrupted ({is_flagged.sum():,} loads)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    # Threshold lines, labelled inside the empty gaps next to the clean cluster.
    axis.axvline(low, color=INK_SECONDARY, lw=1)
    axis.axvline(high, color=INK_SECONDARY, lw=1)
    axis.text(low * 1.04, 300, f"cut-off\n{low}x", color=INK_SECONDARY, fontsize=9, ha="left")
    axis.text(high * 0.96, 300, f"cut-off\n{high}x", color=INK_SECONDARY, fontsize=9, ha="right")
    axis.set_xticks([0.15, 0.25, 0.5, 1, 2, 4, 6])
    axis.set_xticklabels(["0.15x", "0.25x", "0.5x", "1x", "2x", "4x", "6x"])
    axis.set_xlabel("posted_rate / expected price (log scale)")
    axis.set_ylabel("number of loads (log scale)")
    axis.set_title("Corrupted prices form two separate clusters")
    axis.legend(loc="upper right")
    note(axis, "Expected price = median $/mile for the equipment type and distance, scaled by market_index. "
               "No clean load falls between 0.5x-0.84x or 1.21x-2x.")
    return save_figure(figure, "target_ratio_histogram.png")


def plot_daily_price_level(daily_level, daily_market):
    """
    Two stacked panels sharing the date axis:
      top    - daily median price vs the reference price (%), training period
      bottom - daily median market_index, training and prediction periods
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True,
                                         gridspec_kw={"height_ratios": [1.35, 1]})
    start, end = daily_market.index.min(), daily_market.index.max()
    for axis in (top, bottom):
        shade_quarter_end_months(axis, start, end)

    top.plot(daily_level.index, daily_level.to_numpy(), color=BLUE, lw=1.6)
    top.axhline(0, color=AXIS_LINE, lw=0.8)
    top.set_ylabel("price vs reference (%)")
    top.set_title("Prices ramp up through every quarter-end month (shaded) - market_index does not")

    is_training = daily_market.index <= np.datetime64("2025-10-31")
    bottom.plot(daily_market.index[is_training], daily_market[is_training], color=BLUE, lw=1.2,
                label="training period (Jan-Oct)")
    bottom.plot(daily_market.index[~is_training], daily_market[~is_training], color=ORANGE, lw=1.2,
                label="prediction period (Nov-Dec)")
    bottom.set_ylabel("market_index (daily median)")
    bottom.legend(loc="upper right", ncol=2)
    format_month_axis(bottom)
    note(bottom, "Top: daily median of log(actual / reference price) for clean training loads, as %. "
                 "The reference already includes market_index,\nso what remains is a calendar effect. "
                 "Bottom: the weekly zig-zag is a Thursday-high / Sunday-low cycle.", y=-0.2)
    figure.tight_layout()
    return save_figure(figure, "daily_price_level.png")


def plot_quote_signal_by_month(monthly_values):
    """
    Box plot of quote_signal per month (whiskers = 1st-99th percentile).
    monthly_values: list of (label, values, is_prediction_period).
    """
    apply_style()
    figure, axis = plt.subplots(figsize=(10, 4.2))
    positions = np.arange(len(monthly_values))
    for position, (_label, values, is_prediction) in zip(positions, monthly_values):
        colour = ORANGE if is_prediction else BLUE
        axis.boxplot([values], positions=[position], widths=0.55, whis=(1, 99), showfliers=False,
                     patch_artist=True,
                     boxprops={"facecolor": SURFACE, "edgecolor": colour, "linewidth": 1.4},
                     medianprops={"color": colour, "linewidth": 2},
                     whiskerprops={"color": colour, "linewidth": 1.1},
                     capprops={"color": colour, "linewidth": 1.1})
    axis.set_xticks(positions)
    axis.set_xticklabels([label for label, _values, _flag in monthly_values])
    axis.set_ylabel("quote_signal")
    axis.set_title("quote_signal changes regime by month - Nov/Dec look like August")
    axis.plot([], [], color=BLUE, lw=2, label="training months")
    axis.plot([], [], color=ORANGE, lw=2, label="prediction months")
    axis.legend(loc="upper left", ncol=2)
    note(axis, "Boxes: middle 50% of loads; whiskers: 1st-99th percentile. In August - whose distribution "
               "Nov/Dec match - quote_signal has no relationship with price.")
    return save_figure(figure, "quote_signal_by_month.png")


def plot_city_price_levels(city_table):
    """
    Map of city price levels (synthetic coordinates). Seen cities are coloured
    on a blue (cheaper) - grey - red (pricier) scale; the 8 validation-only
    cities are drawn as hollow circles with their estimated (neighbour) level.
    city_table columns: lat, lon, level_pct, is_unseen.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.grid(False)
    colour_map = LinearSegmentedColormap.from_list("price", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH])
    limit = float(np.abs(city_table["level_pct"]).max())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    seen = city_table[~city_table["is_unseen"]]
    unseen = city_table[city_table["is_unseen"]]
    points = axis.scatter(seen["lon"], seen["lat"], c=seen["level_pct"], cmap=colour_map, norm=norm,
                          s=70, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    axis.scatter(unseen["lon"], unseen["lat"], facecolors="none", edgecolors=INK, s=90, linewidths=1.4,
                 zorder=4, label="city only in validation.csv (level estimated from 5 nearest cities)")
    east_edge = city_table["lon"].max() - 8  # labels near the right edge go to the left of the dot
    for city, row in unseen.iterrows():
        on_east_edge = row["lon"] > east_edge
        axis.annotate(f"{city} ({row['level_pct']:+.1f}%)", (row["lon"], row["lat"]),
                      xytext=(-8, 5) if on_east_edge else (8, 5), ha="right" if on_east_edge else "left",
                      textcoords="offset points", fontsize=8.5, color=INK_SECONDARY)
    colour_bar = figure.colorbar(points, ax=axis, shrink=0.8, pad=0.02)
    colour_bar.set_label("city price level vs average (%)", color=INK_SECONDARY)
    colour_bar.outline.set_visible(False)
    axis.set_xlabel("longitude (synthetic)")
    axis.set_ylabel("latitude (synthetic)")
    axis.set_title("City price levels change smoothly across the map")
    axis.legend(loc="upper left")
    note(axis, "Level = average of log(actual / reference price) for loads touching the city, compared with "
               "other loads on the same day.")
    return save_figure(figure, "city_price_levels.png")


# =============================================================================
# Phase 4 / 5 figures
# =============================================================================
def plot_model_comparison(names, mean_mae, std_mae, title, filename):
    """Horizontal bars: mean fold MAE per experiment (lower is better), thin std whiskers."""
    apply_style()
    order = np.argsort(mean_mae)[::-1]  # best at the top
    names = [names[i] for i in order]
    mean_mae = np.asarray(mean_mae)[order]
    std_mae = np.asarray(std_mae)[order]
    figure, axis = plt.subplots(figsize=(9.5, 0.5 * len(names) + 1.4))
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    positions = np.arange(len(names))
    axis.barh(positions, mean_mae, height=0.62, color=BLUE)
    axis.errorbar(mean_mae, positions, xerr=std_mae, fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3)
    for position, value, spread in zip(positions, mean_mae, std_mae):
        axis.text(value + spread + mean_mae.max() * 0.01, position, f"${value:,.1f}", va="center",
                  fontsize=9, color=INK_SECONDARY)
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.set_xlabel("mean MAE over the 4 rolling folds ($, clean labels) - lower is better")
    axis.set_title(title)
    axis.set_xlim(0, (mean_mae + std_mae).max() * 1.15)
    return save_figure(figure, filename)


def plot_feature_importance(importance, title, top_n=20):
    """Horizontal bars of the top features' share of total (tree) importance."""
    apply_style()
    share = (importance / importance.sum()).sort_values(ascending=False).head(top_n)[::-1]
    figure, axis = plt.subplots(figsize=(9, 0.32 * len(share) + 1.3))
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.barh(np.arange(len(share)), 100 * share.to_numpy(), height=0.66, color=BLUE)
    axis.set_yticks(np.arange(len(share)))
    axis.set_yticklabels(share.index)
    axis.set_xlabel("share of total gain in the tree part (%)")
    axis.set_title(title)
    return save_figure(figure, "tree_feature_importance.png")


def plot_group_importance(group_importance, model_name):
    """
    Horizontal bars: how much the Sep-Oct holdout MAE rises when one feature
    group is shuffled (permutation importance). Bigger = the model relies on it more.
    """
    apply_style()
    table = group_importance.sort_values("MAE_increase")
    figure, axis = plt.subplots(figsize=(9.5, 0.45 * len(table) + 1.6))
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    positions = np.arange(len(table))
    values = table["MAE_increase"].to_numpy()
    axis.barh(positions, values, height=0.62, color=BLUE)
    axis.set_xscale("symlog", linthresh=1)
    for position, value, percent in zip(positions, values, table["MAE_increase_%"].to_numpy()):
        axis.text(max(value, 0) * 1.08 + 0.3, position, f"+${value:,.1f} (+{percent:,.0f}%)",
                  va="center", fontsize=9, color=INK_SECONDARY)
    axis.set_yticks(positions)
    axis.set_yticklabels(table["feature_group"])
    axis.set_xlim(min(0, values.min() * 2), values.max() * 4)  # a negative rise would mean the group hurts
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _position: f"${value:,.0f}"))
    axis.set_xlabel("rise in holdout MAE when the group is shuffled ($, log scale)")
    axis.set_title(f"Which feature groups the final model relies on ({model_name})")
    note(axis, f"Permutation importance on the Sep-Oct holdout (model trained on Jan-Aug). "
               f"Baseline MAE ${table['holdout_MAE'].iloc[0]:,.2f}.", y=-0.16)
    return save_figure(figure, "feature_group_importance.png")


def plot_monthly_rate_per_mile(train_monthly, predicted_monthly):
    """
    Average $/mile per month: actual (training, Jan-Oct) vs predicted (Nov-Dec).
    Both Series are indexed by month-start timestamps.
    """
    apply_style()
    figure, axis = plt.subplots(figsize=(9.5, 4))
    axis.plot(train_monthly.index, train_monthly.to_numpy(), color=BLUE, marker="o", markersize=5,
              label="actual (training, clean labels)")
    axis.plot(predicted_monthly.index, predicted_monthly.to_numpy(), color=ORANGE, marker="o", markersize=5,
              label="predicted (validation.csv)")
    axis.set_ylabel("average $ per mile")
    axis.set_title("Average rate per mile by month - actual vs predicted")
    axis.legend(loc="upper left")
    format_month_axis(axis)
    note(axis, "Monthly mean of posted_rate / distance. The mix of distances is similar in every month.")
    return save_figure(figure, "monthly_rate_per_mile.png")


def plot_december_predictions(dates, predicted_rate, market_index):
    """
    Two stacked panels for the fixed December load:
      top    - predicted rate per day
      bottom - the market_index value fed to the model for that day
    """
    apply_style()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True,
                                         gridspec_kw={"height_ratios": [1.4, 1]})
    top.plot(dates, predicted_rate, color=BLUE, marker="o", markersize=4)
    top.set_ylabel("predicted rate ($)")
    top.set_title("December 2025: Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb")
    first, last = predicted_rate[0], predicted_rate[-1]
    top.annotate(f"${first:,.0f}", (dates[0], first), xytext=(-4, 10), textcoords="offset points",
                 fontsize=9, color=INK_SECONDARY, ha="right")
    top.annotate(f"${last:,.0f}", (dates[-1], last), xytext=(4, 10), textcoords="offset points",
                 fontsize=9, color=INK_SECONDARY)

    bottom.plot(dates, market_index, color=AQUA, marker="o", markersize=3.5, lw=1.4)
    bottom.set_ylabel("market_index used")
    bottom.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    note(bottom, "market_index = daily median of validation.csv loads on that date (Thursday high, Sunday low). "
                 "The upward drift is the\nquarter-end ramp learned from March, June and September.", y=-0.28)
    figure.tight_layout()
    return save_figure(figure, "december_predictions_diagnostic.png")
