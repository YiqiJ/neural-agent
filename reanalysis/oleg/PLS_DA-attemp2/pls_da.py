import math
import os
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "oleg_data.parquet"
OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "pls_da"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_SEC = 0.275
# timepoints in seconds relative to stimulus onset
TIMEPOINTS_S = [0.15, 1.0]

# Random seed for shuffling reproducibility
RNG_SEED = 42


@dataclass
class PanelData:
    X_scores: np.ndarray  # shape (n_trials, 2)
    y: np.ndarray         # shape (n_trials,) with values in {-1, +1}
    lda_coef: np.ndarray  # shape (2,) LDA coefficients (w)
    lda_intercept: float  # scalar (b)
    x_thresh_diag: float  # vertical threshold for diagonal discrimination (x = const)


def time_to_bin_index(t_sec: float) -> int:
    """
    Map a time (sec) to the sample index bin that contains this time, assuming bins are
    [k*BIN_SEC, (k+1)*BIN_SEC). For t=0.15 with BIN_SEC=0.275, returns 0.
    For t=1.0, returns floor(1.0/0.275)=3.
    """
    if t_sec < 0:
        raise ValueError("Time must be >= 0 relative to stimulus onset")
    return int(math.floor(t_sec / BIN_SEC))


def read_mouse_timepoint(mouse_id: str, sample_idx: int) -> pd.DataFrame:
    """
    Load rows for a specific mouse and sample_idx efficiently using pyarrow filters.
    Returns a pandas DataFrame with columns: trial_idx, cell_idx, amplitude, behavior.
    """
    filters = [
        ("mouse_id", "==", mouse_id),
        ("sample_idx", "==", sample_idx),
    ]

    try:
        table = pq.read_table(DATA_PATH, filters=filters, columns=[
            "trial_idx", "cell_idx", "amplitude", "behavior"
        ])
    except TypeError:
        # Older pyarrow may not support filters on single-file parquet; fallback to pandas
        df_all = pd.read_parquet(DATA_PATH, columns=[
            "mouse_id", "trial_idx", "cell_idx", "amplitude", "behavior", "sample_idx"
        ])
        table = df_all[(df_all.mouse_id == mouse_id) & (df_all.sample_idx == sample_idx)][
            ["trial_idx", "cell_idx", "amplitude", "behavior"]
        ]
        return table.reset_index(drop=True)

    return table.to_pandas()


def build_trial_by_neuron(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    """
    Given a long-form df with columns trial_idx, cell_idx, amplitude, behavior, build:
      - X_df: pivoted matrix (rows=trial_idx, cols=cell_idx)
      - y: labels mapped to {-1, +1} from behavior {-30, +30}
      - cell_order: list of cell_idx column order
    """
    # Keep only two classes we expect
    df = df[df["behavior"].isin([-30, 30])].copy()
    # Labels per trial (behavior should be consistent per trial)
    trial_labels = (
        df[["trial_idx", "behavior"]].drop_duplicates().set_index("trial_idx")["behavior"]
    )
    # Pivot to trials x cells
    X_df = df.pivot(index="trial_idx", columns="cell_idx", values="amplitude").sort_index()
    # Map behaviors to -1/+1
    y = trial_labels.loc[X_df.index].map({-30: -1, 30: 1}).astype(int).values

    # Fill any missing values (should be rare) with 0.0
    if X_df.isna().any().any():
        X_df = X_df.fillna(0.0)

    cell_order = X_df.columns.tolist()
    return X_df, y, cell_order


def fit_pls_da(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Fit PLSRegression using y in {-1, +1} as the discriminant target.
    Return the X_scores (n_samples x n_components) to use as 2D coordinates.
    """
    # sklearn's PLSRegression supports multi-target; y must be 2D array
    y_cont = y.reshape(-1, 1).astype(float)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=n_comp, scale=True)
    pls.fit(X, y_cont)
    X_scores = pls.x_scores_  # (n_samples, n_components)
    # Ensure shape has 2 columns (pad if necessary)
    if X_scores.shape[1] < 2:
        pad = np.zeros((X_scores.shape[0], 2 - X_scores.shape[1]))
        X_scores = np.hstack([X_scores, pad])
    return X_scores[:, :2]


def fit_linear_boundary(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit LDA on 2D scores to get a linear decision boundary: w^T x + b = 0.
    Returns (w, b).
    """
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(scores, y)
    # For binary, coef_.shape = (1, 2), intercept_.shape=(1,)
    w = lda.coef_.ravel()
    b = float(lda.intercept_.ravel()[0])
    return w, b


def diagonal_discrimination_threshold_x(scores: np.ndarray, y: np.ndarray) -> float:
    """
    A simple diagonal-discrimination-inspired boundary: threshold along x-axis (dim 1)
    at the midpoint of class means on dim-1, yielding a vertical decision line.
    """
    x1 = scores[:, 0]
    mu_neg = float(np.mean(x1[y == -1]))
    mu_pos = float(np.mean(x1[y == +1]))
    return 0.5 * (mu_neg + mu_pos)


def plot_four_panel(mouse_id: str,
                    results: Dict[str, PanelData],
                    out_path: Path) -> None:
    """
    Create a 2x2 figure for the given mouse with panels:
      TL: real @ 0.15s, TR: real @ 1.0s,
      BL: shuf @ 0.15s, BR: shuf @ 1.0s.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)
    fig.suptitle(f"PLS-DA projections: {mouse_id}")

    # Determine shared axis limits across all panels for consistency
    all_scores = []
    for k, pdta in results.items():
        all_scores.append(pdta.X_scores)
    all_scores = np.vstack(all_scores)
    x_min, x_max = np.percentile(all_scores[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_scores[:, 1], [1, 99])
    pad_x = 0.1 * (x_max - x_min + 1e-9)
    pad_y = 0.1 * (y_max - y_min + 1e-9)
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)

    panel_order = [
        ("real_0.15", 0, 0, None),
        ("real_1.00", 0, 1, None),
        ("shuf_0.15", 1, 0, None),
        ("shuf_1.00", 1, 1, None),
    ]

    for key, r, c, title in panel_order:
        ax = axes[r][c]
        pdta = results[key]
        xs = pdta.X_scores
        y = pdta.y

        # Scatter points as crosses
        ax.scatter(xs[y == -1, 0], xs[y == -1, 1], marker='x', color='blue', alpha=0.7, label='Stim -30')
        ax.scatter(xs[y == +1, 0], xs[y == +1, 1], marker='x', color='red', alpha=0.7, label='Stim +30')

        # Green LDA boundary
        w, b = pdta.lda_coef, pdta.lda_intercept
        # Plot line over x-range
        xx = np.linspace(x_lim[0], x_lim[1], 200)
        if abs(w[1]) > 1e-12:
            yy = -(w[0] * xx + b) / w[1]
            ax.plot(xx, yy, color='green', linewidth=2, label='LDA boundary (real/shuf)')
        else:
            # Vertical line
            x0 = -b / (w[0] + 1e-12)
            ax.axvline(x0, color='green', linewidth=2, label='LDA boundary (vertical)')

        # Black vertical diagonal-discrimination boundary (x = threshold)
        ax.axvline(pdta.x_thresh_diag, color='black', linewidth=2, linestyle='-', label='Diagonal discrim.')

        if title is None:
            # Pretty title from key, e.g., 'real_1.00' -> 'Real @ 1.0 s'
            kind, tstr = key.split("_")
            t_pretty = f"{float(tstr):.1f}"
            title = ("Real" if kind == "real" else "Shuffled") + f" @ {t_pretty} s"
        ax.set_title(title)
        ax.set_xlabel('PLS dim-1')
        ax.set_ylabel('PLS dim-2')
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        # Add legend only to top-right to avoid clutter
        if r == 0 and c == 1:
            ax.legend(loc='best', fontsize=8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def trial_shuffle_within_stimulus(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Shuffle amplitudes across trials independently for each (cell_idx, behavior) group.
    Returns a new DataFrame with the same shape and columns; 'amplitude' is permuted
    within each group. Implemented with vectorized per-group NumPy shuffles for speed.
    """
    out = df.copy()
    amp = out["amplitude"].to_numpy(copy=True)
    cell = out["cell_idx"].to_numpy()
    beh = out["behavior"].to_numpy()
    # Map behavior {-30, 30} -> {0,1}
    beh01 = (beh > 0).astype(np.int16)
    # Numeric group id: 2*cell + beh01 (safe within int64)
    grp_ids = (cell.astype(np.int64) << 1) + beh01.astype(np.int64)
    order = np.argsort(grp_ids, kind='mergesort')
    grp_sorted = grp_ids[order]
    amp_sorted = amp[order]
    # Find group boundaries
    boundaries = np.flatnonzero(np.diff(grp_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [grp_sorted.size]))
    # Shuffle each contiguous segment
    for s, e in zip(starts, ends):
        if e - s > 1:
            rng.shuffle(amp_sorted[s:e])
    # Undo sorting
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    amp = amp_sorted[inv_order]
    out.loc[:, "amplitude"] = amp
    return out.reset_index(drop=True)


def prepare_panel(df_tp: pd.DataFrame) -> PanelData:
    # Build matrix and labels
    X_df, y, _ = build_trial_by_neuron(df_tp)
    X = X_df.to_numpy(dtype=float)
    # PLS-DA to 2D
    scores = fit_pls_da(X, y, n_components=2)
    # LDA boundary on scores
    w, b = fit_linear_boundary(scores, y)
    # Diagonal discrimination threshold along x-axis
    xthr = diagonal_discrimination_threshold_x(scores, y)
    return PanelData(X_scores=scores, y=y, lda_coef=w, lda_intercept=b, x_thresh_diag=xthr)


def main():
    parser = argparse.ArgumentParser(description="PLS-DA reproduction for neural data")
    parser.add_argument("--mouse", type=str, default=None, help="Run only this mouse ID (e.g., Mouse_L347)")
    args = parser.parse_args()
    print("[INFO] Starting PLS-DA reproduction...")
    print(f"[INFO] Data path: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    # Identify mice quickly by reading distinct values
    print("[INFO] Reading distinct mouse IDs...")
    # Read minimal columns to get mice
    tmeta = pq.read_table(DATA_PATH, columns=["mouse_id"]).to_pandas()
    mice = sorted(tmeta["mouse_id"].unique().tolist())
    if args.mouse is not None:
        if args.mouse not in mice:
            raise ValueError(f"Requested mouse {args.mouse} not found; available: {mice}")
        mice = [args.mouse]
    print(f"[INFO] Found mice: {mice}")

    rng = np.random.default_rng(RNG_SEED)

    # Map desired timepoints to sample indices
    tp_bins = {f"{t:.2f}": time_to_bin_index(t) for t in TIMEPOINTS_S}
    print(f"[INFO] Timepoint to bin indices: {tp_bins}")

    for mouse_id in mice:
        print(f"\n[INFO] Processing mouse: {mouse_id}")
        results: Dict[str, PanelData] = {}

        # Real data panels
        for t_key, sidx in tp_bins.items():
            print(f"[INFO]  Loading real data at {t_key}s (sample_idx={sidx})...")
            df_tp = read_mouse_timepoint(mouse_id, sidx)
            n_rows = len(df_tp)
            n_trials = df_tp["trial_idx"].nunique()
            n_cells = df_tp["cell_idx"].nunique()
            print(f"[INFO]   rows={n_rows}, trials={n_trials}, cells={n_cells}")
            # Prepare panel
            pdta = prepare_panel(df_tp)
            # Log boundary summary
            print(f"[INFO]   LDA boundary w={pdta.lda_coef}, b={pdta.lda_intercept:.4f}; diag-x-thresh={pdta.x_thresh_diag:.4f}")
            key = f"real_{t_key}"
            results[key] = pdta

        # Shuffled data panels
        for t_key, sidx in tp_bins.items():
            print(f"[INFO]  Loading and shuffling data at {t_key}s (sample_idx={sidx})...")
            df_tp = read_mouse_timepoint(mouse_id, sidx)
            df_shuf = trial_shuffle_within_stimulus(df_tp, rng)
            n_rows = len(df_shuf)
            n_trials = df_shuf["trial_idx"].nunique()
            n_cells = df_shuf["cell_idx"].nunique()
            print(f"[INFO]   shuffled rows={n_rows}, trials={n_trials}, cells={n_cells}")
            pdta = prepare_panel(df_shuf)
            print(f"[INFO]   Shuf LDA boundary w={pdta.lda_coef}, b={pdta.lda_intercept:.4f}; diag-x-thresh={pdta.x_thresh_diag:.4f}")
            key = f"shuf_{t_key}"
            results[key] = pdta

        # Plot per mouse
        out_path = OUT_DIR / f"pls_da_{mouse_id}.png"
        print(f"[INFO]  Plotting and saving to: {out_path}")
        plot_four_panel(mouse_id, results, out_path)
        print(f"[INFO]  Done mouse {mouse_id}")

    print("\n[INFO] All mice processed. Figures saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
