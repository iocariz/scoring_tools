# MR Risk Inference Methodology

## Overview

MR (Model Reserve) is a post-production monitoring mechanism that validates risk estimates on a separate, more recent observation period. It compares predictions from the initial training period (main period) with actual outcomes observed in the MR window, enabling early detection of model degradation.

The core challenge: the MR period is recent, so 6-month (H6) outcomes may not be fully mature. The pipeline addresses this by using 3-month (H3) outcomes — which mature faster — and extrapolating them to the H6 horizon using the relationship observed in the main period.

## Risk Metric Formulas

All risk metrics derive from the same base formula, differing only by horizon and multiplier:

```
b2_ever_h6 = multiplier_h6 × (todu_30ever_h6 / todu_amt_pile_h6)

b2_ever_h3 = multiplier_h3 × (todu_30ever_h3 / todu_amt_pile_h3)
```

| Component          | Description                                        | Default |
|--------------------|----------------------------------------------------|---------|
| `todu_30ever_hN`   | Count of 30+ day delinquent accounts (N-month window) | —       |
| `todu_amt_pile_hN` | Total exposure amount (N-month window)             | —       |
| `multiplier_h6`    | H6 scaling factor                                  | 7       |
| `multiplier_h3`    | H3 scaling factor                                  | 4       |

Division by zero yields NaN (handled downstream via `np.nan_to_num`).

## Data Flow

### 1. Period Splitting

The preprocessed data is split by date into two non-overlapping windows:

- **Main period** (`date_ini_book_obs` → `date_fin_book_obs`): training/calibration data with fully mature H6 outcomes.
- **MR period** (`date_ini_book_obs_mr` → `date_fin_book_obs_mr`): recent data where H6 may be immature but H3 is available.

### 2. Per-Bin Aggregation

Both periods are aggregated by bin combination (e.g., octroi bin × EFX bin). For each bin:

**Main period:**
```
b2_main    = multiplier_h6 × Σ(todu_30ever_h6) / Σ(todu_amt_pile_h6)
b2_main_h3 = multiplier_h3 × Σ(todu_30ever_h3) / Σ(todu_amt_pile_h3)   [if H3 available]
n_obs_main = count of booked records
```

**MR period:**
```
b2_mr      = multiplier_h6 × Σ(todu_30ever_h6) / Σ(todu_amt_pile_h6)
b2_mr_h3   = multiplier_h3 × Σ(todu_30ever_h3) / Σ(todu_amt_pile_h3)   [mature accounts only]
n_obs_mr   = count of booked records
n_obs_mr_h3 = count of mature H3 records (may be < n_obs_mr)
```

For H3 aggregation in the MR period, only accounts with non-null H3 columns are included — this filters out immature accounts that haven't reached the 3-month mark.

### 3. Risk Source Selection

For each bin, the pipeline selects a risk estimate following a strict priority:

#### Priority 1 — H3 Extrapolation

**Conditions:** H3 data exists AND `n_obs_mr_h3 ≥ min_obs` AND `b2_main_h3` is non-zero.

```
h6_h3_ratio = b2_main / b2_main_h3        (main-period scaling factor)
b2_ever_h6  = b2_mr_h3 × h6_h3_ratio      (extrapolated MR risk)
```

**Rationale:** Mature 3-month MR outcomes, scaled by the main period's H6/H3 relationship, provide an early estimate of 6-month risk before H6 outcomes fully mature.

#### Priority 2 — MR Observed

**Conditions:** `n_obs_mr ≥ min_obs` AND H3 extrapolation was not triggered.

```
b2_ever_h6 = b2_mr                         (direct H6 observation)
```

**Rationale:** Sufficient MR observations provide direct H6 evidence.

#### Priority 3 — Main-Period Imputed

**Conditions:** `n_obs_mr < min_obs` (sparse MR data) but bin exists in main period.

```
b2_ever_h6 = b2_main                       (training-period risk)
```

**Rationale:** Historical main-period risk serves as fallback when MR data is too sparse.

#### Priority 4 — Model Fallback

**Conditions:** Bin absent from main period AND `n_obs_mr < min_obs`.

```
b2_ever_h6 = NaN → filled by risk model inference (calculate_B2)
```

**Rationale:** For entirely new bin combinations, the trained risk model provides the estimate.

Each bin's selected source is logged in the `risk_source` column of the comparison output for auditability.

### 4. Revenue Prediction

For the MR period, `todu_amt_pile_h6` (exposure) is predicted using a regression model trained on the main period:

```
todu_amt_pile_h6_bin = reg_todu_amt_pile.predict(oa_amt_bin)
```

This is then pro-rated back to account level:

```
todu_amt_pile_h6_account = (todu_amt_pile_h6_bin / oa_amt_bin) × oa_amt_account
```

The risk numerator is derived via inverse formula:

```
todu_30ever_h6 = b2_ever_h6 × todu_amt_pile_h6 / multiplier_h6
```

## H3 Extrapolation — Worked Example

Consider bin (octroi=2, efx=5):

**Main period (fully mature):**
- `todu_30ever_h6` = 100, `todu_amt_pile_h6` = 1000 → `b2_main` = 7 × 100/1000 = 70%
- `todu_30ever_h3` = 45, `todu_amt_pile_h3` = 900 → `b2_main_h3` = 4 × 45/900 = 20%
- **H6/H3 ratio:** 70% / 20% = 3.5

**MR period (H3 mature, H6 immature):**
- `todu_30ever_h3` = 8, `todu_amt_pile_h3` = 160 → `b2_mr_h3` = 4 × 8/160 = 20%
- **Extrapolated H6 risk:** 20% × 3.5 = **70%**
- **Risk source:** `h3_extrapolated`

The MR 3-month outcomes (20%), scaled by the main period's natural H6/H3 relationship (3.5×), estimate 6-month MR risk at 70%.

## Comparison Diagnostics

For every bin, the pipeline computes drift metrics:

```
b2_delta     = b2_mr − b2_main
b2_delta_pct = (b2_delta / b2_main) × 100

[if H3 available:]
b2_delta_h3     = b2_mr_h3 − b2_main_h3
b2_delta_pct_h3 = (b2_delta_h3 / b2_main_h3) × 100
h6_h3_ratio     = b2_main / b2_main_h3
```

Output is saved as `mr_risk_comparison{suffix}.csv` containing per-bin: `b2_main`, `b2_mr`, `n_obs_main`, `n_obs_mr`, `b2_ever_h6_tmp`, `risk_source`, `b2_delta`, `b2_delta_pct`, `mr_production`, and optional H3 columns.

## Risk Production Summary

The function `calculate_metrics_from_cuts` applies the optimal cutoff solution to MR data and produces a summary table with four rows:

| Row           | Definition                                    |
|---------------|-----------------------------------------------|
| **Actual**    | All booked accounts                           |
| **Swap-in**   | Rejected (repesca) accounts that pass the cut |
| **Swap-out**  | Booked accounts that fail the cut             |
| **Optimum**   | Actual − Swap-out + Swap-in                   |

For each row, both production (`oa_amt_h0`) and risk (`b2_ever_h6`, and optionally `b2_ever_h3`) are computed, enabling a direct comparison of what the optimal policy would have achieved on MR-period data.

## Stability Validation (PSI/CSI)

Population Stability Index compares main vs. MR distributions:

```
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
```

| PSI Range       | Interpretation                    |
|-----------------|-----------------------------------|
| < 0.10          | Stable                            |
| 0.10 – 0.25    | Moderate drift — investigate      |
| ≥ 0.25          | Unstable — action required        |

CSI (Characteristic Stability Index) uses the same formula applied to categorical variable distributions.

## Configuration

All MR parameters are set in `config.toml` (or per-segment in `segments.toml`):

| Parameter               | Type    | Default | Description                                           |
|-------------------------|---------|---------|-------------------------------------------------------|
| `use_mr_outcomes`       | bool    | false   | Enable hybrid MR risk computation                     |
| `date_ini_book_obs_mr`  | string  | null    | MR window start date                                  |
| `date_fin_book_obs_mr`  | string  | null    | MR window end date                                    |
| `multiplier`            | float   | 7.0     | H6 risk multiplier                                    |
| `multiplier_h3`         | float   | 4.0     | H3 risk multiplier                                    |
| `mr_min_obs_per_bin`    | int     | 30      | Minimum observations per bin to use MR-observed risk   |

Both MR dates must be provided together (all-or-nothing validation).
