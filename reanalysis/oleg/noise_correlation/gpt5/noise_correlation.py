"""
Noise Correlation Analysis in Neural Populations

This script reads the dataset at data/oleg_data.parquet and performs:
1) Dataset summary per mouse: number of cells, number of trials, behavior types, and trial length stats.
2) Reproduction of the figure described in the caption:
   - Compute cell mean responses to the two stimuli (behavior == +30 and -30) per mouse.
   - Select top 10% most active cells by sqrt(<r_A>^2 + <r_B>^2).
   - Compute pairwise noise correlations across trials after removing per-stimulus mean responses.
   - Split cell pairs into positively vs negatively correlated mean responses groups based on
	 sign((rA_i - rB_i) * (rA_j - rB_j)).
   - Run two-tailed KS tests comparing the two distributions per mouse and overall.
   - Plot overlaid histograms and save p-value table.

Outputs:
- reanalysis_oleg/noise_correlation_summary.csv
- reanalysis_oleg/noise_corr_pvalues_per_mouse.csv
- reanalysis_oleg/combined_noise_correlation_histogram_all_mice_reprod.png
- reanalysis_oleg/noise_corr_hist_<{mouse_id}>.png (per mouse)

Assumptions:
- Trial-level response is the mean amplitude across time samples (sample_idx) within a (mouse, behavior, trial_idx, cell_idx).
- Noise correlations computed on trial-by-trial residuals after subtracting per-stimulus mean response for each cell.
- For grouping of pairs, positivity/negativity is based on the direction of tuning difference between stimuli.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "oleg_data.parquet"
OUT_DIR = Path(__file__).resolve().parent


def read_data(parquet_path: Path) -> pd.DataFrame:
	"""Read the parquet file with minimal overhead.

	Returns a DataFrame with columns:
	- mouse_id (object)
	- sample_idx (int32)
	- cell_idx (int32)
	- amplitude (float32)
	- trial_idx (int32)
	- behavior (int16, values {+30, -30})
	"""
	import pyarrow.parquet as pq

	table = pq.read_table(parquet_path)
	df = table.to_pandas(split_blocks=True, self_destruct=True)
	# Normalize dtypes to keep memory lower
	df["mouse_id"] = df["mouse_id"].astype("category")
	return df


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute dataset summary per mouse.

	Returns DataFrame with columns:
	- mouse_id
	- n_cells
	- n_trials
	- behavior_types (comma-separated)
	- trial_length_mean, trial_length_std, trial_length_min, trial_length_max
	"""
	# cells per mouse
	cells_per_mouse = (
		df.groupby("mouse_id")["cell_idx"].nunique().rename("n_cells").reset_index()
	)

	# trials per mouse
	trials_per_mouse = (
		df.groupby("mouse_id")["trial_idx"].nunique().rename("n_trials").reset_index()
	)

	# behavior types
	behav_types = (
		df.groupby("mouse_id")["behavior"]
		.apply(lambda s: ",".join(map(str, sorted(pd.unique(s)))))
		.rename("behavior_types")
		.reset_index()
	)

	# trial lengths: number of samples per (mouse, trial_idx)
	trial_lengths = (
		df.groupby(["mouse_id", "trial_idx"])['sample_idx']
		.nunique()
		.rename("samples_per_trial")
		.reset_index()
	)
	tl_stats = (
		trial_lengths.groupby("mouse_id")["samples_per_trial"]
		.agg(trial_length_mean="mean",
			 trial_length_std="std",
			 trial_length_min="min",
			 trial_length_max="max")
		.reset_index()
	)

	summary = cells_per_mouse.merge(trials_per_mouse, on="mouse_id")
	summary = summary.merge(behav_types, on="mouse_id")
	summary = summary.merge(tl_stats, on="mouse_id")
	# sort by mouse id string for readability
	summary = summary.sort_values("mouse_id").reset_index(drop=True)
	return summary


def compute_cell_mean_responses(df_mouse: pd.DataFrame) -> pd.DataFrame:
	"""Compute per-cell mean responses to each stimulus for a given mouse.

	Returns a wide DataFrame indexed by cell_idx with columns r_{stim} for each behavior level.
	"""
	# trial-level mean per (behavior, trial, cell)
	trial_resp = (
		df_mouse.groupby(["behavior", "trial_idx", "cell_idx"], observed=True)["amplitude"]
		.mean()
		.rename("trial_mean")
		.reset_index()
	)

	cell_means = (
		trial_resp.groupby(["behavior", "cell_idx"], observed=True)["trial_mean"]
		.mean()
		.rename("mean_resp")
		.reset_index()
	)

	# Pivot to wide: columns per behavior level
	wide = cell_means.pivot(index="cell_idx", columns="behavior", values="mean_resp").fillna(0.0)
	# Rename columns for clarity: behavior values like -30, 30
	wide.columns = [f"r_{int(c)}" for c in wide.columns]
	return wide


def select_top_active_cells(cell_means: pd.DataFrame, top_frac: float = 0.10) -> List[int]:
	"""Select top fraction of cells by sqrt(rA^2 + rB^2).

	Expects columns r_-30 and r_30 to exist; missing columns are treated as 0.
	Returns a list of cell indices.
	"""
	# Ensure both columns exist
	for col in ("r_30", "r_-30"):
		if col not in cell_means.columns:
			cell_means[col] = 0.0

	magnitude = np.sqrt(np.square(cell_means["r_30"]) + np.square(cell_means["r_-30"]))
	k = max(1, int(math.ceil(len(magnitude) * top_frac)))
	top_idx = np.argpartition(-magnitude.values, k - 1)[:k]
	return cell_means.index.values[top_idx].tolist()


def pair_sign_groups(cell_means: pd.DataFrame) -> Dict[Tuple[int, int], str]:
	"""For each cell pair, determine if their mean responses are positively or negatively correlated.

	Using sign of (rA - rB) alignment. Returns dict mapping (i, j) with i < j to 'pos' or 'neg'.
	Pairs with zero product are omitted.
	"""
	rA = cell_means.get("r_30", pd.Series(0.0, index=cell_means.index))
	rB = cell_means.get("r_-30", pd.Series(0.0, index=cell_means.index))
	d = (rA - rB)
	pairs: Dict[Tuple[int, int], str] = {}
	idx = cell_means.index.values
	# Vectorized approach is complex; use nested loops for clarity since k is modest (top 10%).
	for a in range(len(idx)):
		for b in range(a + 1, len(idx)):
			prod = float(d.iloc[a] * d.iloc[b])
			if prod > 0:
				pairs[(int(idx[a]), int(idx[b]))] = "pos"
			elif prod < 0:
				pairs[(int(idx[a]), int(idx[b]))] = "neg"
			# else, skip if exactly zero (ambiguous)
	return pairs


def compute_noise_correlation_for_mouse(df_mouse: pd.DataFrame, selected_cells: List[int]) -> Tuple[List[float], List[float]]:
	"""Compute noise correlations for a single mouse, returning two lists of r-values:
	- pos_pairs: pairs with positive mean-response alignment
	- neg_pairs: pairs with negative mean-response alignment

	Residuals are computed per stimulus by subtracting each cell's per-stimulus mean across trials,
	then correlations are computed across the concatenated trials from both stimuli.
	"""
	# Build trial-level response matrix per stimulus and compute residuals
	# First, compute per-cell per-stimulus mean across trials (for residual centering)
	cell_means = compute_cell_mean_responses(df_mouse)
	# Restrict to selected cells subset
	cell_means = cell_means.loc[cell_means.index.intersection(selected_cells)].copy()
	if cell_means.empty or len(cell_means.index) < 2:
		return [], []

	# Determine pair grouping based on tuning alignment
	pair_groups = pair_sign_groups(cell_means)
	if not pair_groups:
		return [], []

	# Prepare residual matrices per stimulus
	pos_vals: List[float] = []
	neg_vals: List[float] = []

	# Compute trial-level mean matrix for each stimulus
	for_behavior = {}
	for stim in df_mouse["behavior"].unique():
		sub = df_mouse[df_mouse["behavior"] == stim]
		pivot = (
			sub.groupby(["trial_idx", "cell_idx"], observed=True)["amplitude"]
			.mean()
			.rename("trial_mean")
			.reset_index()
			.pivot(index="trial_idx", columns="cell_idx", values="trial_mean")
		)
		# Keep only selected cells and align column order
		pivot = pivot.reindex(columns=cell_means.index, fill_value=np.nan)
		# Center by per-stimulus mean for each cell (computed from cell_means)
		col_mean = cell_means.get(f"r_{int(stim)}", pd.Series(0.0, index=cell_means.index))
		pivot = pivot.subtract(col_mean, axis=1)
		for_behavior[int(stim)] = pivot

	# Concatenate trials from both stimuli (rows stack), aligning columns
	matrices = [for_behavior[k] for k in sorted(for_behavior.keys())]
	if not matrices:
		return [], []
	X = pd.concat(matrices, axis=0, ignore_index=True)

	# Compute pairwise correlations using pairwise complete observations
	# Using pandas corr is efficient and handles NaNs pairwise
	corr_mat = X.corr(method="pearson", min_periods=2)

	# Extract upper triangle pairs
	cols = list(corr_mat.columns)
	col_pos = {int(c): i for i, c in enumerate(cols)}
	for (i, j), grp in pair_groups.items():
		# If either column missing or insufficient data, skip
		if i not in col_pos or j not in col_pos:
			continue
		r = corr_mat.iat[col_pos[i], col_pos[j]]
		if pd.isna(r):
			continue
		if grp == "pos":
			pos_vals.append(float(r))
		else:
			neg_vals.append(float(r))

	return pos_vals, neg_vals


def analyze_and_plot(df: pd.DataFrame) -> None:
	# Summary
	summary = dataset_summary(df)
	summary_out = OUT_DIR / "noise_correlation_summary.csv"
	summary.to_csv(summary_out, index=False)
	print("Summary saved to:", summary_out)
	print(summary.to_string(index=False))

	all_pos: List[float] = []
	all_neg: List[float] = []
	per_mouse_counts = []
	pvals = []

	mice = list(summary["mouse_id"].astype(str))
	for m in mice:
		print(f"\nProcessing mouse {m} ...")
		df_mouse = df[df["mouse_id"].astype(str) == m]
		# Select top 10% active cells
		cell_means = compute_cell_mean_responses(df_mouse)
		top_cells = select_top_active_cells(cell_means, top_frac=0.10)
		print(f"Selected top 10% cells: {len(top_cells)} / {cell_means.shape[0]} cells")

		pos_vals, neg_vals = compute_noise_correlation_for_mouse(df_mouse, top_cells)
		print(f"Pairs: pos={len(pos_vals)}, neg={len(neg_vals)}")
		all_pos.extend(pos_vals)
		all_neg.extend(neg_vals)
		per_mouse_counts.append({
			"mouse_id": m,
			"n_cells_top10": len(top_cells),
			"n_pairs_pos": len(pos_vals),
			"n_pairs_neg": len(neg_vals),
		})

		# KS test for this mouse
		if len(pos_vals) > 1 and len(neg_vals) > 1:
			stat, p = ks_2samp(pos_vals, neg_vals, alternative="two-sided", mode="auto")
		else:
			stat, p = (np.nan, np.nan)
		pvals.append({"mouse_id": m, "ks_stat": stat, "p_value": p})

		# Plot per-mouse histogram
		fig, ax = plt.subplots(figsize=(6, 4))
		bins = np.linspace(-0.2, 0.4, 31)
		# Red: similarly tuned (positive alignment); Blue: differently tuned (negative alignment)
		ax.hist(pos_vals, bins=bins, histtype='step', linewidth=2.0, label="Similarly tuned", density=False, color="#d62728")
		ax.hist(neg_vals, bins=bins, histtype='step', linewidth=2.0, label="Differently tuned", density=False, color="#1f77b4")
		ax.set_title(f"Noise correlations: {m}\nKS p={p:.2e}" if not np.isnan(p) else f"Noise correlations: {m}")
		ax.set_xlabel("Correlation coefficient")
		ax.set_ylabel("Number of cell pairs")
		ax.set_xlim(-0.2, 0.4)
		ax.legend(frameon=False)
		ax.grid(True, alpha=0.2)
		fig.tight_layout()
		out_file = OUT_DIR / f"noise_corr_hist_{m}.png"
		fig.savefig(out_file, dpi=150)
		plt.close(fig)
		print("Saved per-mouse histogram:", out_file)

	# Save counts and p-values per mouse
	counts_df = pd.DataFrame(per_mouse_counts)
	pvals_df = pd.DataFrame(pvals)
	pv_out = OUT_DIR / "noise_corr_pvalues_per_mouse.csv"
	pvals_df.to_csv(pv_out, index=False)
	print("P-values saved to:", pv_out)
	print(pvals_df.to_string(index=False))

	# Overall KS
	overall_p = np.nan
	overall_stat = np.nan
	if len(all_pos) > 1 and len(all_neg) > 1:
		overall_stat, overall_p = ks_2samp(all_pos, all_neg, alternative="two-sided", mode="auto")

	# Combined histogram
	fig, ax = plt.subplots(figsize=(7.5, 5))
	bins = np.linspace(-0.2, 0.4, 31)
	ax.hist(all_pos, bins=bins, histtype='step', linewidth=2.2, label=f"Similarly tuned (n={len(all_pos)})", density=False, color="#d62728")
	ax.hist(all_neg, bins=bins, histtype='step', linewidth=2.2, label=f"Differently tuned (n={len(all_neg)})", density=False, color="#1f77b4")
	title = "Noise correlations (Top 10% cells across mice)"
	if not np.isnan(overall_p):
		title += f"\nOverall KS p={overall_p:.2e}"
	ax.set_title(title)
	ax.set_xlabel("Correlation coefficient")
	ax.set_ylabel("Number of cell pairs")
	ax.set_xlim(-0.2, 0.4)
	ax.legend(frameon=False)
	ax.grid(True, alpha=0.2)

	# Add p-value table
	try:
		tbl = pvals_df.copy()
		tbl["ks_stat"] = tbl["ks_stat"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
		tbl["p_value"] = tbl["p_value"].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
		table_data = [tbl.columns.tolist()] + tbl.values.tolist()
		from matplotlib.table import Table

		# Place a small table below plot
		# Using a basic text box to avoid layout complexity
		text_lines = ["Per-mouse KS p-values:"] + [
			f"- {row['mouse_id']}: p={row['p_value']} (stat={row['ks_stat']})" for _, row in tbl.iterrows()
		]
		ax.text(1.02, 0.5, "\n".join(text_lines), transform=ax.transAxes, va="center", ha="left", fontsize=9)
	except Exception:
		pass

	fig.tight_layout()
	out_file = OUT_DIR / "combined_noise_correlation_histogram_all_mice_reprod.png"
	fig.savefig(out_file, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print("Saved combined histogram:", out_file)


def main():
	print("Loading data from:", DATA_PATH)
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Parquet not found: {DATA_PATH}")
	df = read_data(DATA_PATH)
	print("Data shape:", df.shape)
	print("Columns:", df.columns.tolist())
	analyze_and_plot(df)


if __name__ == "__main__":
	main()

