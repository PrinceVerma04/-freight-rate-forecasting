# Loom script (about 2 minutes 40 seconds)

**Before you record** - open these tabs, in this order, so you only switch tabs while talking:

1. The GitHub repository (README)
2. `reports/figures/target_ratio_histogram.png`
3. `reports/figures/daily_price_level.png`
4. `src/data_cleaning.py`
5. `src/config.py` (scrolled to `ROLLING_FOLDS`)
6. `reports/figures/model_comparison.png`
7. `src/models.py` (scrolled to `class RidgeThenLightGBMModel`)
8. `run_pipeline.py`
9. `scorer_results/candidate_december.png`

The spoken part is about 400 words - speak calmly. **SHOW** lines are what is on screen; the quoted
lines are what you say.

---

### 1. The problem (0:00 - 0:15)

**SHOW:** the README on GitHub.

> Hi, I'm Prince. This is my solution to Spotter's freight rate challenge: predicting the price of
> twelve thousand loads in November and December, from forty-eight thousand labelled loads from
> January to October. Every load is in the future, so I treated it as a forecasting problem.

### 2. What I found in the data (0:15 - 0:40)

**SHOW:** `target_ratio_histogram.png`

> Distance drives most of the price, then equipment and weight. To check the labels, I compared each
> load with a simple expected price. Clean loads sit within twenty percent of it, but 677 loads -
> 1.4 percent - are at a fraction or a multiple of it. Those are corrupted prices.

**SHOW:** `daily_price_level.png`

> Prices also climb about four percent through every quarter-end month, which the market index
> doesn't explain - and December is a quarter-end month.

### 3. How I fixed the data (0:40 - 1:05)

**SHOW:** `src/data_cleaning.py` - scroll slowly past the function names.

> Cleaning has one function per issue. Negative weights were sign flips, so I made them positive.
> Missing weights get the equipment median, capped weights get a flag, and missing market index
> values come from the same day. Corrupted prices are removed from training and from scoring.
> Distances that looked too long were genuine seventy-mile minimum loads, so I kept them. Everything
> is learned from the training rows only.

### 4. How I split and validated (1:05 - 1:30)

**SHOW:** `src/config.py` at `ROLLING_FOLDS`.

> I didn't use a random split. I used four rolling monthly folds: train January to June, test July;
> add July, test August; and so on up to October. Every decision is based on the average error over
> those four months. A random split looks eight percent better, because its test loads share days
> with training loads - which is impossible when you forecast December.

### 5. The model and why (1:30 - 2:05)

**SHOW:** `model_comparison.png`

> I compared a baseline, Ridge regression, LightGBM, CatBoost and ensembles. The tree models were
> precise, but they missed badly in October, the start of a new quarter, because trees can't extend a
> trend. Ridge can, but it's less precise.

**SHOW:** `src/models.py` at `class RidgeThenLightGBMModel`.

> So my final model combines them: Ridge predicts first, then LightGBM learns only Ridge's errors.
> The trend stays linear, and the trees add the detail. Its error is about thirty-two dollars per
> load, 1.36 percent - two thirds below the baseline. I also dropped quote_signal: it broke the
> August fold, and November and December look like August.

### 6. Code and December chart (2:05 - 2:30)

**SHOW:** `run_pipeline.py` - scroll through the phase functions.

> Everything runs with one command, run_pipeline.py, with one function per phase, ending with
> Spotter's scorer.

**SHOW:** `candidate_december.png`

> The December chart rises from 834 to 878 dollars: a weekly market cycle, plus the quarter-end ramp
> learned from March, June and September.

### 7. Close (2:30 - 2:40)

**SHOW:** the README again.

> The report covers the full validation and the limitations - mainly that one partial year of data
> can't teach holiday seasonality. Thanks for watching.
