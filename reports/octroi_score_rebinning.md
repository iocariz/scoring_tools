# Octroi score (`score_rf`) re-binning — findings & recommendation

**Date:** 2026-06-17  **Scope:** ad-hoc analysis for the management request to consolidate the
octroi score grid from 20 bins to ~10 groups.
**Basis:** through-the-door demand €, ecom + inf-tel.
**Status:** analysis only — nothing wired into the production config yet.

---

## 1. Executive summary

- The request was to split `score_rf` into ~10 groups with **similar population** per group, reusing
  the existing 20-bin edges (merging, and — once confirmed — dividing the dominant bin).
- On the data, the heavy bin is **#4 `[0.9470597, 0.9654058)` = 16.6 % of €** (not "#3" — that was an
  off-by-one against an essentially **empty leading bin #1**, which holds €0.1 M / 0 booked).
- A single new edge inside bin #4 plus merging of the thin tail bins produces a well-balanced grid.
- **Key finding:** `score_rf` carries a genuine **mid-band risk reversal** — bins 5→6→7 have
  b2_ever_h6 **4.86 % → 5.25 % → 5.27 %** (risk *rises* as the score improves). This creates an
  unavoidable tension: you can have **balanced €** *or* **strictly monotone risk**, not both, in that
  band. Strict monotonicity forces those three bins into one 25.6 % lump.
- **Discriminance is not the constraint:** every candidate grouping retains **98.8 – 99.6 %** of the
  score's Gini. The ceiling is the score itself (Gini ≈ **0.30** vs `early_bad`), and the mid-band
  reversal is a minor `score_rf` ranking weakness, not a binning artefact.

### Recommendation
**Adopt the K = 12 scheme** (§5). It gives the tightest € balance (every group 6.4 – 9.9 %), shrinks the
former 25.6 % lump back to 8.3 %, keeps b2 monotone apart from the one intrinsic mid-band bump, and
retains 99.4 % of the score's Gini. Use only **one new edge** (`0.9591544`, the €-median of bin #4);
all other boundaries are existing edges.

---

## 2. Data & method

| Item | Value |
|---|---|
| Source column | `score_rf` (octroi) |
| Demand file | `data/demanda_ecom_inftel_out.sas7bdat` |
| Date window | 2024-06-01 → 2025-05-01 (12 monthly cohorts) |
| Demand applications | 515,890 |
| Through-the-door € (`oa_amt`) | €755.7 M |
| Population weight | **€** (through-the-door `oa_amt`) |
| Risk metric | `b2_ever_h6` = 7·Σ`todu_30ever_h6` / Σ`todu_amt_pile_h6` (booked, realized H6) |
| Discriminance | Gini = 2·AUC−1 vs `early_bad` (89,505 booked, 3.06 % bad-rate) |

**Method.** Candidate groups are contiguous merges of the 20 given bins, chosen by an exact dynamic
program that minimises Σ(group_€ − total/K)². Oversized bins are divided at €-weighted quantiles of
`score_rf` within the bin (new edges). The risk-monotone variant uses Pool-Adjacent-Violators (isotonic)
merging so pooled b2 is strictly decreasing.

---

## 3. Starting point — the current 20 bins

- **Dominant bin #4** `[0.9470597, 0.9654058)`: 16.6 % of €, b2 5.79 %, ≈1.7× a 10 % target.
- **Empty bin #1** `[-inf, 0.8894938)`: €0.1 M, 0 booked (source of the "bin 3" mis-count).
- **Highest risk** at the low-score end (bins 2–3, b2 ≈ 7.5 %), falling to ≈0.3 % at the top.
- **Risk reversals** (b2 rises with score) at bin boundaries **2–3, 5–7, 12–14, 17–18** — the 5–7 one
  is the material one.

See `score_rf_balance_k12.png` (before/after € share + b2; reversals marked black).

---

## 4. The three schemes considered

| Scheme | K | b2 strictly monotone? | € balance (std) | Max € group | New edges | Gini retention |
|---|---|---|---|---|---|---|
| €-balanced (split bin #4) | 10 | ✗ (mid-band bump) | 0.020 | 13.7 % | 1 | 98.8 % |
| Risk-monotone (isotonic) | 10 | ✓ strict | 0.065 | **25.6 %** | 0 | 98.8 % |
| **K = 12 (recommended)** | 12 | ✗ (one bump) | **0.011** | 9.9 % | 1 | 99.4 % |
| K = 14 | 14 | ✗ (two bumps) | 0.019 | 9.9 % | 1 | 99.6 % |

- The **risk-monotone** scheme is the only strictly-monotone option, but the 5–7 reversal forces a
  25.6 % lump — poor € balance.
- Going to **12 / 14 groups** splits bin #4 and lets the tail breathe, so every group lands near target;
  the only cost is the small intrinsic mid-band bump reappearing.

---

## 5. Recommended scheme — K = 12

| grp | range | € share | b2_ever_h6 |
|----|----|----|----|
| 1 | [-inf, 0.9470597) | 9.7 % | 7.51 % |
| 2 | [0.9470597, **0.9591544**) | 8.3 % | 6.50 % |
| 3 | [**0.9591544**, 0.9654058) | 8.3 % | 5.13 % |
| 4 | [0.9654058, 0.9703869) | 8.4 % | 4.86 % |
| 5 | [0.9703869, 0.9744170) | 8.9 % | 5.25 % ↑ |
| 6 | [0.9744170, 0.9777791) | 8.3 % | 5.27 % ↑ |
| 7 | [0.9777791, 0.9806289) | 9.9 % | 4.85 % |
| 8 | [0.9806289, 0.9830639) | 7.3 % | 3.60 % |
| 9 | [0.9830639, 0.9851669) | 6.4 % | 3.15 % |
| 10 | [0.9851669, 0.9902284) | 9.6 % | 2.41 % |
| 11 | [0.9902284, 0.9943728) | 7.6 % | 2.22 % |
| 12 | [0.9943728, +inf) | 7.2 % | 1.00 % |

€ band 6.4 – 9.9 % (std 0.011). b2 monotone except the ↑ at groups 5–6 (4.86 → 5.25 → 5.27, ≈ flat).
The former dominant lump (old bins 5–7, 25.6 % under strict monotonicity) is now three ~8 % groups.

```toml
octroi_bins = [-inf, 0.9470597, 0.9591544, 0.9654058, 0.9703869, 0.9744170, 0.9777791, 0.9806289, 0.9830639, 0.9851669, 0.9902284, 0.9943728, inf]
```

### Conversion map: old bin → new group (K=12)

| Old bin(s) | Old score range | → New group | Action |
|---|---|---|---|
| 1, 2, 3 | [-inf, 0.9470597) | **1** | merge (incl. empty bin 1) |
| 4 | [0.9470597, 0.9654058) | **2 & 3** | **split** at 0.9591544 |
| 5 | [0.9654058, 0.9703869) | 4 | keep |
| 6 | [0.9703869, 0.9744170) | 5 | keep |
| 7 | [0.9744170, 0.9777791) | 6 | keep |
| 8 | [0.9777791, 0.9806289) | 7 | keep |
| 9 | [0.9806289, 0.9830639) | 8 | keep |
| 10 | [0.9830639, 0.9851669) | 9 | keep |
| 11, 12, 13 | [0.9851669, 0.9902284) | **10** | merge |
| 14, 15, 16 | [0.9902284, 0.9943728) | **11** | merge |
| 17, 18, 19, 20 | [0.9943728, +inf) | **12** | merge |

Per-old-bin (explicit): 1→1, 2→1, 3→1, **4→2 & 3 (split)**, 5→4, 6→5, 7→6, 8→7, 9→8, 10→9,
11→10, 12→10, 13→10, 14→11, 15→11, 16→11, 17→12, 18→12, 19→12, 20→12.

Only old bin **4 straddles two new groups** (it is the one bin that gets divided); every other old bin
maps wholly into a single new group. Net change: 20 → 12 groups via **9 merges + 1 split**.

---

## 6. Discriminance (Gini vs `early_bad`)

| scheme | groups | Gini | retention |
|----|----|----|----|
| continuous `score_rf` | — | 0.2991 | 100.0 % |
| 20 given bins | 20 | 0.2973 | 99.4 % |
| K = 10 €-balanced | 10 | 0.2956 | 98.8 % |
| K = 10 risk-monotone | 10 | 0.2956 | 98.8 % |
| **K = 12** | 12 | 0.2974 | 99.4 % |
| K = 14 | 14 | 0.2978 | 99.6 % |

1. **Binning is essentially lossless** — all schemes keep 98.8 – 99.6 % of the score's Gini, so the
   group count and the mid-band reversal cost almost nothing in discriminance (those bins carry similar
   risk, so mis-ordering them barely moves the AUC).
2. **The ceiling is the score.** `score_rf` Gini ≈ 0.30 is modest; the 5–7 reversal is a minor real
   ranking weakness in the score — a note for the scoring/risk team, not fixable by re-binning.

---

## 7. Caveats & governance

- **Population basis = €** (`oa_amt`, through-the-door). An application-count basis gives the same
  dominant bin and a near-identical split point (€-median 0.9591544 vs count-median 0.9591087).
- **Risk is realized booked** (b2_ever_h6 on the booked subset of each bin); rejected applications carry
  no outcome. Gini is vs `early_bad` on the 89,505 booked with a matured flag.
- **The active production config does NOT use these edges** — it uses `max_bins = 10` quantile binning on
  `score_rf`. Switching to a fixed `octroi_bins` array changes the optimisation grid and requires a full
  pipeline re-run plus the usual model-validation sign-off (M5).
- The mid-band reversal is worth a separate look (vintage/mix effect vs a true `score_rf` weakness).

---

## 8. Appendix — all candidate `octroi_bins`

```toml
# K=10, €-balanced (split bin #4)
octroi_bins = [-inf, 0.9470597, 0.9591544, 0.9654058, 0.9703869, 0.9744170, 0.9777791, 0.9806289, 0.9851669, 0.9916906, inf]

# K=10, risk-monotone (strict; one 25.6% group)
octroi_bins = [-inf, 0.9470597, 0.9654058, 0.9777791, 0.9806289, 0.9830639, 0.9851669, 0.9870236, 0.9916906, 0.9943728, inf]

# K=12  (RECOMMENDED)
octroi_bins = [-inf, 0.9470597, 0.9591544, 0.9654058, 0.9703869, 0.9744170, 0.9777791, 0.9806289, 0.9830639, 0.9851669, 0.9902284, 0.9943728, inf]

# K=14
octroi_bins = [-inf, 0.9470597, 0.9591544, 0.9654058, 0.9703869, 0.9744170, 0.9777791, 0.9806289, 0.9830639, 0.9851669, 0.9870236, 0.9902284, 0.9930733, 0.9956056, inf]
```

### Artifacts (regenerated under `--output-dir`, gitignored)
| File | Contents |
|---|---|
| `score_rf_balance_k12.png` | Before (20 bins) vs K=12 balanced — € share + b2 |
| `score_rf_groups_k10/k12/k14.csv` | per-group tables for each K |
| `score_rf_groups_monotone.csv` | risk-monotone (isotonic) scheme |
| `score_rf_bins_before.csv` | the 20 candidate bins (€, count, b2) |
| `score_rf_crosstab_k12.csv` | old bin → new group € mapping |
| `score_rf_report.md` / `score_rf_deck.pptx` | auto-generated report / deck |

### Reproduce
```bash
# bare run reproduces this analysis (defaults = octroi/score_rf); --deck adds the PPTX
uv run python scripts/score_rebin.py \
  --output-dir output/octroi_rebin --title "Octroi re-bin" --date-tag 2026-06-17 --deck
```
The single reusable CLI (`scripts/score_rebin.py`, + `score_rebin_deck.py` for `--deck`) replaces
the earlier one-off scripts; every input (data / score / weight / risk / target / date / segment /
edges / K) is overridable for a different population.
