import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-mouse summary including cells, trials, behavior types, and trial length stats."""
    rows = []
    for m in df["mouse_id"].unique():
        sub = df[df["mouse_id"] == m]
        n_cells = sub["cell_idx"].nunique()
        n_trials = sub["trial_idx"].nunique()
        behaviors = sorted(pd.unique(sub["behavior"]))
        # trial length = number of unique sample_idx per trial
        trial_lengths = (
            sub.groupby("trial_idx")["sample_idx"].nunique().rename("n_samples")
        )
        rows.append(
            {
                "mouse_id": m,
                "n_cells": int(n_cells),
                "n_trials": int(n_trials),
                "behaviors": behaviors,
                "trial_len_min": int(trial_lengths.min()),
                "trial_len_q25": float(trial_lengths.quantile(0.25)),
                "trial_len_med": float(trial_lengths.median()),
                "trial_len_q75": float(trial_lengths.quantile(0.75)),
                "trial_len_max": int(trial_lengths.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_trial_means(df_m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    For a single mouse's data, compute per (cell_idx, trial_idx) mean amplitude and return:
      - wide matrix Y with index trial_idx and columns cell_idx (values = mean amplitude)
      - per-trial behavior Series aligned to Y.index
    """
    # Mean amplitude per cell per trial
    mean_df = (
        df_m.groupby(["cell_idx", "trial_idx"], observed=True)["amplitude"].mean().astype(np.float32)
    )
    mean_df = mean_df.reset_index()

    # Behavior per trial
    beh_per_trial = (
        df_m.groupby("trial_idx", observed=True)["behavior"].first().astype(np.int16)
    )

    # Pivot to trials x cells
    Y = mean_df.pivot(index="trial_idx", columns="cell_idx", values="amplitude").sort_index()

    # Align behaviors to Y.index just in case
    beh = beh_per_trial.reindex(Y.index)
    return Y, beh


def top_active_cells(Y: pd.DataFrame, beh: pd.Series, top_frac: float = 0.10) -> List[int]:
    """
    Select top fraction of cells by activity magnitude sqrt(rA^2 + rB^2),
    where rA and rB are the mean responses to the two stimuli (beh values).
    Returns list of selected cell_idx (column names) as ints.
    """
    beh_values = pd.unique(beh.dropna())
    if len(beh_values) != 2:
        raise ValueError(f"Expected exactly 2 behavior/stimulus values, got {beh_values}")
    s1, s2 = sorted(beh_values)

    Y1 = Y.loc[beh == s1]
    Y2 = Y.loc[beh == s2]
    r1 = Y1.mean(axis=0)  # per cell
    r2 = Y2.mean(axis=0)
    mag = np.sqrt(r1**2 + r2**2)

    k = max(1, int(np.ceil(len(mag) * top_frac)))
    sel_cells = mag.sort_values(ascending=False).head(k).index.to_list()
    return sel_cells


def compute_noise_correlations(Y: pd.DataFrame, beh: pd.Series) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute noise correlations between all cell pairs, and split into pairs with positively vs. negatively
    correlated mean responses to the two stimuli.

    Returns (pos_corrs, neg_corrs, n_cells, n_trials)
    """
    beh_values = pd.unique(beh.dropna())
    if len(beh_values) != 2:
        raise ValueError("Noise correlation requires exactly 2 stimuli/behaviors")
    s1, s2 = sorted(beh_values)

    # Compute per-stimulus means per cell
    Y1 = Y.loc[beh == s1]
    Y2 = Y.loc[beh == s2]
    r1 = Y1.mean(axis=0)
    r2 = Y2.mean(axis=0)

    # Residuals: subtract per-stimulus mean from each trial
    R1 = Y1 - r1
    R2 = Y2 - r2
    R = pd.concat([R1, R2], axis=0)

    # Drop any all-NaN columns (shouldn't happen but safe)
    R = R.dropna(axis=1, how="all")
    # Fill remaining NaNs per column with 0 mean (if some cells missing some trials)
    R = R.fillna(0.0)

    # Pairwise Pearson correlation among cells
    X = R.values.astype(np.float32, copy=False)
    if X.shape[1] < 2:
        return np.array([]), np.array([]), X.shape[1], R.shape[0]

    # Normalize columns
    X = X - X.mean(axis=0, keepdims=True)
    denom = X.std(axis=0, ddof=1, keepdims=True)
    denom[denom == 0] = 1.0
    Xn = X / denom
    corr = np.matmul(Xn.T, Xn) / (Xn.shape[0] - 1)

    # Classify pairs by tuning sign
    pref_sign = np.sign((r1 - r2).reindex(R.columns).fillna(0.0).values)
    # indices of upper triangle pairs
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    pair_corrs = corr[iu, ju]

    same_pref = pref_sign[iu] * pref_sign[ju] > 0
    opp_pref = pref_sign[iu] * pref_sign[ju] < 0

    pos_corrs = pair_corrs[same_pref]
    neg_corrs = pair_corrs[opp_pref]
    return pos_corrs, neg_corrs, n, R.shape[0]


def ks_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS test. Use scipy if available, else fallback to numpy-based implementation."""
    try:
        from scipy.stats import ks_2samp  # type: ignore

        res = ks_2samp(a, b, alternative="two-sided", method="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Fallback implementation (asymptotic p-value)
        a = np.asarray(a)
        b = np.asarray(b)
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan
        data_all = np.concatenate([a, b])
        cdf1 = np.searchsorted(np.sort(a), data_all, side="right") / a.size
        cdf2 = np.searchsorted(np.sort(b), data_all, side="right") / b.size
        d = np.max(np.abs(cdf1 - cdf2))
        n1, n2 = a.size, b.size
        en = np.sqrt(n1 * n2 / (n1 + n2))
        # Marsaglia et al. approximation
        lam = (en + 0.12 + 0.11 / en) * d
        # Kolmogorov distribution Q_KS(lam)
        # Q_KS(x) ~ 2 sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 x^2)
        terms = [2 * ((-1) ** (k - 1)) * np.exp(-2 * (k**2) * (lam**2)) for k in range(1, 101)]
        p = float(np.clip(np.sum(terms), 0.0, 1.0))
        return float(d), p


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "oleg_data.parquet"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading data from {data_path} ...")
    df = pd.read_parquet(data_path)

    # Basic dtypes
    df["mouse_id"] = df["mouse_id"].astype("category")
    for c in ["sample_idx", "cell_idx", "trial_idx", "behavior"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    if "amplitude" in df:
        df["amplitude"] = pd.to_numeric(df["amplitude"], downcast="float")

    # 1) Dataset summary
    summary_df = summarize_dataset(df)
    print("\n=== Dataset summary per mouse ===")
    print(summary_df.to_string(index=False))

    # Save summary
    summary_csv = out_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved dataset summary to {summary_csv}")

    # 2) Noise correlation analysis for top 10% most active cells
    mice = list(df["mouse_id"].cat.categories)
    all_pos: List[np.ndarray] = []
    all_neg: List[np.ndarray] = []
    per_mouse_stats: List[Dict] = []

    for m in mice:
        print(f"\n--- Processing {m} ---")
        df_m = df[df["mouse_id"] == m].copy()
        Y, beh = compute_trial_means(df_m)
        print(
            f"Trials: {Y.shape[0]} | Cells: {Y.shape[1]} | Behaviors: {sorted(pd.unique(beh.dropna()))}"
        )

        # Select top 10% active cells
        try:
            sel = top_active_cells(Y, beh, top_frac=0.10)
        except ValueError as e:
            print(f"Skipping {m} due to error: {e}")
            continue

        Y_sel = Y[sel]
        pos_corrs, neg_corrs, n_cells, n_trials = compute_noise_correlations(Y_sel, beh)
        print(
            f"Selected cells: {n_cells} | trials used: {n_trials} | +pairs: {len(pos_corrs)} | -pairs: {len(neg_corrs)}"
        )

        # KS per mouse
        D, p = ks_test(pos_corrs, neg_corrs)
        print(f"KS test (pos vs neg): D={D:.4f}, p={p:.3e}")

        per_mouse_stats.append(
            {
                "mouse_id": m,
                "n_cells_selected": int(n_cells),
                "+pairs": int(len(pos_corrs)),
                "-pairs": int(len(neg_corrs)),
                "KS_D": float(D),
                "KS_p": float(p),
            }
        )

        all_pos.append(pos_corrs)
        all_neg.append(neg_corrs)

        # Release memory
        del df_m, Y, beh, Y_sel
        gc.collect()

    if not all_pos or not all_neg:
        print("No data collected for noise correlations. Exiting.")
        return

    pos_all = np.concatenate(all_pos)
    neg_all = np.concatenate(all_neg)

    # Combined KS across all mice (for reference)
    D_all, p_all = ks_test(pos_all, neg_all)
    print(f"\n=== Combined across mice ===\nKS (pos vs neg): D={D_all:.4f}, p={p_all:.3e}")
    print(
        f"Total +pairs: {len(pos_all)} | Total -pairs: {len(neg_all)} | Total selected cells (sum across mice): {sum(d['n_cells_selected'] for d in per_mouse_stats)}"
    )

    # Plot histogram
    import matplotlib.pyplot as plt

    bins = np.linspace(min(pos_all.min(), neg_all.min()), max(pos_all.max(), neg_all.max()), 50)
    plt.figure(figsize=(8, 5))
    plt.hist(pos_all, bins=bins, alpha=0.6, label="Positively tuned pairs", color="#1f77b4")
    plt.hist(neg_all, bins=bins, alpha=0.6, label="Negatively tuned pairs", color="#ff7f0e")
    plt.axvline(np.median(pos_all), color="#1f77b4", ls="--", lw=1)
    plt.axvline(np.median(neg_all), color="#ff7f0e", ls="--", lw=1)
    plt.xlabel("Noise correlation coefficient")
    plt.ylabel("Number of pairs")
    plt.title("Noise correlations: top 10% active cells, grouped by tuning agreement")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "combined_noise_correlation_histogram_all_mice.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved histogram figure to {fig_path}")

    # Build results table per the requested schema
    # headers: comparison, test, N-value, p-values for each mouse, num. cells, num mice
    # We'll report per-mouse p-values and overall counts
    res_table = {
        "comparison": [
            "pairwise noise correlations, positively vs. negatively tuned cells, top 10% active cells"
        ],
        "test": ["Kolmogorov–Smirnov"],
        "N-value": [f"+pairs={len(pos_all)}, -pairs={len(neg_all)}"],
        "num. cells": [int(sum(d["n_cells_selected"] for d in per_mouse_stats))],
        "num mice": [len(per_mouse_stats)],
    }
    # Add per-mouse p-values
    for d in per_mouse_stats:
        res_table[f"p({d['mouse_id']})"] = [d["KS_p"]]

    results_df = pd.DataFrame(res_table)
    table_path = out_dir / "noise_correlation_results_table.csv"
    results_df.to_csv(table_path, index=False)
    print(f"Saved results table to {table_path}")

    # Also save a more detailed per-mouse stats CSV
    detailed_df = pd.DataFrame(per_mouse_stats)
    detailed_path = out_dir / "per_mouse_stats.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Saved per-mouse stats to {detailed_path}")


if __name__ == "__main__":
    main()
