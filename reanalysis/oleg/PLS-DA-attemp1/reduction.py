import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BIN_SEC = 0.275
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "oleg_data.parquet"))


@dataclass
class DataSummary:
	n_mice: int
	cells_per_mouse: Dict[str, int]
	trials_per_mouse_total: Dict[str, int]
	trials_per_mouse_by_behavior: pd.DataFrame
	behavior_types: List[int]
	trial_bins_stats: pd.DataFrame
	trial_seconds_stats: pd.DataFrame


def read_summary() -> DataSummary:
	"""Read global summary using pandas (fast enough here) to satisfy the user's request."""
	# Use pandas to read minimal columns for summary
	cols = ["mouse_id", "cell_idx", "trial_idx", "sample_idx", "behavior"]
	df = pd.read_parquet(DATA_PATH, columns=cols)

	mice = sorted(df["mouse_id"].unique())
	n_mice = len(mice)

	cells_per_mouse = (
		df[["mouse_id", "cell_idx"]]
		.drop_duplicates()
		.groupby("mouse_id")["cell_idx"].nunique()
		.to_dict()
	)

	trials_df = df[["mouse_id", "trial_idx", "behavior"]].drop_duplicates()
	trials_per_mouse_total = (
		trials_df.groupby("mouse_id")["trial_idx"].nunique().to_dict()
	)
	trials_per_mouse_by_behavior = (
		trials_df.groupby(["mouse_id", "behavior"])["trial_idx"].nunique().unstack(fill_value=0)
	)
	behavior_types = sorted(trials_df["behavior"].unique().tolist())

	trial_bins = df[["mouse_id", "trial_idx", "sample_idx"]].drop_duplicates()
	bins_per_trial = trial_bins.groupby(["mouse_id", "trial_idx"])["sample_idx"].nunique()
	trial_bins_stats = bins_per_trial.groupby("mouse_id").agg(["min", "max", "median", "mean"]).round(3)
	trial_seconds_stats = (trial_bins_stats[["min", "max", "median", "mean"]] * BIN_SEC).round(3)

	return DataSummary(
		n_mice=n_mice,
		cells_per_mouse=cells_per_mouse,
		trials_per_mouse_total=trials_per_mouse_total,
		trials_per_mouse_by_behavior=trials_per_mouse_by_behavior,
		behavior_types=behavior_types,
		trial_bins_stats=trial_bins_stats,
		trial_seconds_stats=trial_seconds_stats,
	)


def _arrow_filtered_table(mouse_id: str, sample_idx: int, columns: List[str]) -> pd.DataFrame:
	"""Load a filtered slice from parquet using pyarrow.dataset for efficiency."""
	dataset = ds.dataset(DATA_PATH, format="parquet")
	filt = (ds.field("mouse_id") == mouse_id) & (ds.field("sample_idx") == sample_idx)
	table = dataset.to_table(columns=columns, filter=filt)
	return table.to_pandas()


def build_trial_matrix(mouse_id: str, sample_idx: int) -> Tuple[pd.DataFrame, pd.Series]:
	"""Return X (trials x cells) and y (behavior) for a given mouse and time bin.

	X rows are aligned to sorted trial_idx, columns to sorted cell_idx.
	"""
	cols = ["trial_idx", "cell_idx", "amplitude", "behavior", "mouse_id", "sample_idx"]
	df = _arrow_filtered_table(mouse_id, sample_idx, cols)

	# Pivot into trials x cells
	pivot = df.pivot_table(index="trial_idx", columns="cell_idx", values="amplitude", aggfunc="mean")
	pivot = pivot.sort_index().sort_index(axis=1)

	# Labels per trial
	labels = df.drop_duplicates(subset=["trial_idx"]).set_index("trial_idx").loc[pivot.index, "behavior"]

	return pivot, labels


def choose_bin_for_time_sec(t: float) -> int:
	"""Choose sample_idx whose bin center is closest to the target time."""
	# bin center at (k + 0.5) * BIN_SEC; solve for k ≈ t/BIN_SEC - 0.5
	k = int(round(t / BIN_SEC - 0.5))
	return max(0, min(k, 13))  # observed bins are 0..13


def subsample_balanced(X: pd.DataFrame, y: pd.Series, per_class: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.Series]:
	classes = sorted(y.unique())
	idx_sel: List[int] = []
	for c in classes:
		idx_c = y[y == c].index
		n_take = min(per_class, len(idx_c))
		chosen = rng.choice(idx_c, size=n_take, replace=False)
		idx_sel.extend(chosen)
	idx_sel = sorted(idx_sel)
	return X.loc[idx_sel], y.loc[idx_sel]


def fit_pls_and_project(X: pd.DataFrame, y: pd.Series, n_components: int = 2) -> Tuple[PLSRegression, np.ndarray, np.ndarray, StandardScaler]:
	# Standardize features (per neuron) for stability
	scaler = StandardScaler(with_mean=True, with_std=True)
	Xs = scaler.fit_transform(X.values)
	# Encode labels to {-1, +1}
	y_enc = np.where(y.values.astype(int) > 0, 1.0, -1.0)
	pls = PLSRegression(n_components=n_components, scale=False)  # already standardized
	pls.fit(Xs, y_enc)
	Z = pls.transform(Xs)  # (n_trials, 2)
	return pls, Z, y_enc, scaler


def fit_linear_boundary(Z: np.ndarray, y_enc: np.ndarray) -> Tuple[np.ndarray, float]:
	"""Fit a linear boundary in 2D using logistic regression and return (w, b) for w^T z + b = 0."""
	clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")
	clf.fit(Z, (y_enc > 0).astype(int))
	w = clf.coef_[0]
	b = clf.intercept_[0]
	return w, b


def project_boundary_line(ax, w: np.ndarray, b: float, color: str, label: str, xmin: float, xmax: float, linestyle: str = "-"):
	# For 2D: w0 * x + w1 * y + b = 0 -> y = -(w0*x + b)/w1
	xs = np.linspace(xmin, xmax, 100)
	if abs(w[1]) < 1e-8:
		# vertical line
		x0 = -b / (w[0] + 1e-12)
		ax.plot([x0, x0], [ax.get_ylim()[0], ax.get_ylim()[1]], color=color, linestyle=linestyle, label=label)
	else:
		ys = -(w[0] * xs + b) / w[1]
		ax.plot(xs, ys, color=color, linestyle=linestyle, label=label)


def diagonal_discrimination_threshold(Z: np.ndarray, y_enc: np.ndarray) -> float:
	"""Axis-aligned (ignore correlations) boundary on Z[:,0]: midpoint of class means along first axis."""
	z1 = Z[y_enc > 0, 0]
	z2 = Z[y_enc < 0, 0]
	return 0.5 * (z1.mean() + z2.mean())


def plot_plsda_for_times(mouse_id: str, times_sec: List[float], per_class: int = 220, seed: int = 0, out_path: str | None = None):
	rng = np.random.default_rng(seed)
	ncols = len(times_sec)
	fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
	if ncols == 1:
		axes = [axes]

	logs: List[str] = []
	for i, t in enumerate(times_sec):
		ax = axes[i]
		k = choose_bin_for_time_sec(t)
		logs.append(f"Time {t:.3f}s -> sample_idx={k} (bin center={(k+0.5)*BIN_SEC:.3f}s)")

		X, y = build_trial_matrix(mouse_id, k)
		logs.append(f"Loaded matrix for {mouse_id} at bin {k}: X shape={X.shape}, trials per class: {y.value_counts().to_dict()}")

		Xb, yb = subsample_balanced(X, y, per_class=per_class, rng=rng)
		logs.append(f"Subsampled to per-class={min(per_class, y.value_counts().min())}: Xb={Xb.shape}")

		pls, Z, y_enc, scaler = fit_pls_and_project(Xb, yb, n_components=2)
		logs.append("Fitted PLSRegression with 2 components; projected to Z (n x 2)")

		# Plot trials
		m1 = y_enc > 0
		m2 = ~m1
		ax.scatter(Z[m1, 0], Z[m1, 1], c="#1f77b4", marker="x", label="stim +", alpha=0.7)
		ax.scatter(Z[m2, 0], Z[m2, 1], c="#d62728", marker="x", label="stim -", alpha=0.7)

		# Real boundary (green)
		w_real, b_real = fit_linear_boundary(Z, y_enc)
		xmin, xmax = np.percentile(Z[:, 0], [1, 99])
		project_boundary_line(ax, w_real, b_real, color="green", label="real boundary", xmin=xmin, xmax=xmax)

		# Trial-shuffled boundary (orange): shuffle trials independently per neuron (in X-space), then project via fitted PLS
		Xs = scaler.transform(Xb.values)
		Xs_shuff = Xs.copy()
		for j in range(Xs_shuff.shape[1]):
			rng.shuffle(Xs_shuff[:, j])
		Z_shuff = pls.transform(Xs_shuff)
		w_shuf, b_shuf = fit_linear_boundary(Z_shuff, y_enc)
		project_boundary_line(ax, w_shuf, b_shuf, color="orange", label="trial-shuffled", xmin=xmin, xmax=xmax, linestyle="--")

		# Diagonal discrimination (black vertical): threshold on first axis only
		thr = diagonal_discrimination_threshold(Z, y_enc)
		ax.axvline(thr, color="black", linestyle=":", label="diagonal discr.")

		ax.set_title(f"{mouse_id} @ {t:.2f}s (n={int(m1.sum())} vs {int(m2.sum())})")
		ax.set_xlabel("PLS comp 1")
		if i == 0:
			ax.set_ylabel("PLS comp 2")
		ax.legend(loc="best", fontsize=9)
		ax.grid(True, alpha=0.2)

	fig.suptitle("Neural ensemble responses in PLS-DA space (real vs trial-shuffled)")
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])

	if out_path is None:
		out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, "plsda_real_vs_shuffled.png")

	fig.savefig(out_path, dpi=200)
	plt.close(fig)

	return out_path, logs


def plot_plsda_on_axes(mouse_id: str, times_sec: List[float], axes: List[plt.Axes], per_class: int = 220, seed: int = 0) -> List[str]:
	"""Plot PLS-DA panels for a single mouse over provided axes (one axis per time)."""
	rng = np.random.default_rng(seed)
	logs: List[str] = []
	for i, (t, ax) in enumerate(zip(times_sec, axes)):
		k = choose_bin_for_time_sec(t)
		logs.append(f"Time {t:.3f}s -> sample_idx={k} (bin center={(k+0.5)*BIN_SEC:.3f}s)")
		X, y = build_trial_matrix(mouse_id, k)
		logs.append(f"Loaded matrix for {mouse_id} at bin {k}: X shape={X.shape}, trials per class: {y.value_counts().to_dict()}")
		Xb, yb = subsample_balanced(X, y, per_class=per_class, rng=rng)
		logs.append(f"Subsampled to per-class={min(per_class, y.value_counts().min())}: Xb={Xb.shape}")
		pls, Z, y_enc, scaler = fit_pls_and_project(Xb, yb, n_components=2)
		logs.append("Fitted PLSRegression with 2 components; projected to Z (n x 2)")

		# Points
		m1 = y_enc > 0
		m2 = ~m1
		ax.scatter(Z[m1, 0], Z[m1, 1], c="#1f77b4", marker="x", label="stim +", alpha=0.7, s=16)
		ax.scatter(Z[m2, 0], Z[m2, 1], c="#d62728", marker="x", label="stim -", alpha=0.7, s=16)

		# Boundaries
		w_real, b_real = fit_linear_boundary(Z, y_enc)
		xmin, xmax = np.percentile(Z[:, 0], [1, 99])
		project_boundary_line(ax, w_real, b_real, color="green", label="real boundary", xmin=xmin, xmax=xmax)

		Xs = scaler.transform(Xb.values)
		Xs_shuff = Xs.copy()
		for j in range(Xs_shuff.shape[1]):
			rng.shuffle(Xs_shuff[:, j])
		Z_shuff = pls.transform(Xs_shuff)
		w_shuf, b_shuf = fit_linear_boundary(Z_shuff, y_enc)
		project_boundary_line(ax, w_shuf, b_shuf, color="orange", label="trial-shuffled", xmin=xmin, xmax=xmax, linestyle="--")

		thr = diagonal_discrimination_threshold(Z, y_enc)
		ax.axvline(thr, color="black", linestyle=":", label="diagonal discr.")

		ax.set_title(f"{mouse_id} @ {t:.2f}s (n+={int(m1.sum())}, n-={int(m2.sum())})", fontsize=10)
		ax.set_xlabel("PLS comp 1")
		if i == 0:
			ax.set_ylabel("PLS comp 2")
		ax.grid(True, alpha=0.2)
	return logs


def build_trial_matrix_all_mice(sample_idx: int) -> Tuple[pd.DataFrame, pd.Series]:
	"""Build trials x neurons matrix by concatenating trials from all mice at a given time bin.

	- Rows: (mouse_id, trial_idx)
	- Columns: f"{mouse_id}::{cell_idx}" (unique per neuron across all mice)
	- Values: amplitude
	"""
	dataset = ds.dataset(DATA_PATH, format="parquet")
	filt = (ds.field("sample_idx") == sample_idx)
	cols = ["mouse_id", "trial_idx", "cell_idx", "amplitude", "behavior", "sample_idx"]
	table = dataset.to_table(columns=cols, filter=filt)
	df = table.to_pandas()
	pivots: List[pd.DataFrame] = []
	labels: List[pd.Series] = []
	for mid, d in df.groupby("mouse_id"):
		pv = d.pivot_table(index="trial_idx", columns="cell_idx", values="amplitude", aggfunc="mean").sort_index().sort_index(axis=1)
		pv.columns = [f"{mid}::{c}" for c in pv.columns]
		pv.index = pd.MultiIndex.from_product([[mid], pv.index], names=["mouse_id", "trial_idx"])
		pivots.append(pv)
		lab = d.drop_duplicates(subset=["trial_idx"]).set_index("trial_idx").sort_index()["behavior"]
		lab.index = pd.MultiIndex.from_product([[mid], lab.index], names=["mouse_id", "trial_idx"])
		labels.append(lab)
	X = pd.concat(pivots, axis=0, sort=False)
	y = pd.concat(labels).loc[X.index]
	return X, y


def standardize_ignore_nan(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Standardize each feature using only non-NaN rows; set NaN positions to 0 after standardizing.

	Returns (Xs, means, stds) where Xs is ndarray and means/stds are per-column arrays.
	"""
	A = X.to_numpy(dtype=float)
	n, d = A.shape
	means = np.zeros(d)
	stds = np.ones(d)
	Xs = np.zeros_like(A)
	for j in range(d):
		col = A[:, j]
		m = ~np.isnan(col)
		if not m.any():
			# no valid entries, keep zeros
			means[j] = 0.0
			stds[j] = 1.0
			continue
		mu = col[m].mean()
		sigma = col[m].std(ddof=0)
		if sigma < 1e-12:
			sigma = 1.0
		means[j] = mu
		stds[j] = sigma
		Xs[m, j] = (col[m] - mu) / sigma
		# where not m -> already 0
	return Xs, means, stds


def main():
	warnings.filterwarnings("ignore", category=UserWarning)

	# 1) Data summary
	print("[Step 1] Reading data summary from:", DATA_PATH)
	summary = read_summary()
	print("Successfully read data.")
	print("Number of mice:", summary.n_mice)
	print("Cells per mouse:")
	for k, v in sorted(summary.cells_per_mouse.items()):
		print(f"  {k}: {v}")
	print("Trials per mouse (total):")
	for k, v in sorted(summary.trials_per_mouse_total.items()):
		print(f"  {k}: {v}")
	print("Trials per mouse by behavior:")
	print(summary.trials_per_mouse_by_behavior)
	print("Behavior types:", summary.behavior_types)
	print("Trial length (bins) stats per mouse:")
	print(summary.trial_bins_stats)
	print("Trial length (seconds) stats per mouse:")
	print(summary.trial_seconds_stats)

	# 2) Reproduce figure for ALL mice (rows = mice, columns = two timepoints)
	mice_sorted = sorted(summary.trials_per_mouse_total, key=summary.trials_per_mouse_total.get, reverse=True)
	times = [0.15, 1.0]
	nrows, ncols = len(mice_sorted), len(times)
	fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows), sharex=False, sharey=False)
	if nrows == 1:
		axes = np.array([axes])
	for r, m in enumerate(mice_sorted):
		row_axes = axes[r]
		logs = plot_plsda_on_axes(m, times_sec=times, axes=list(row_axes), per_class=220, seed=42)
		for line in logs:
			print(f"LOG[{m}]:", line)
		# place a compact legend only on the first row, first column
		if r == 0:
			row_axes[0].legend(loc="best", fontsize=9)
	fig.suptitle("Neural ensemble responses in PLS-DA space (all mice, real vs trial-shuffled)")
	fig.tight_layout(rect=[0, 0.03, 1, 0.97])
	out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
	os.makedirs(out_dir, exist_ok=True)
	out_all = os.path.join(out_dir, "plsda_all_mice.png")
	fig.savefig(out_all, dpi=200)
	plt.close(fig)
	print("Saved multi-mouse figure to:", out_all)

	# 3) Mouse L347: stimulus-conditional trial-shuffled analysis averaged over 100 subsets
	print("\n[Step 3] Mouse_L347 trial-shuffled analysis (stimulus-conditional, 100 repeats)...")
	mouse_id = "Mouse_L347"
	times = [0.15, 1.0]
	fig, axes = plt.subplots(1, len(times), figsize=(6*len(times), 5), sharey=True)
	if not isinstance(axes, np.ndarray):
		axes = np.array([axes])

	# Helper: build full X,y once per bin for shuffling against full set
	def build_full(mouse_id: str, k: int):
		X_full, y_full = build_trial_matrix(mouse_id, k)
		# ensure index sorted and int typed for mapping
		X_full = X_full.sort_index()
		y_full = y_full.loc[X_full.index]
		return X_full, y_full

	def subsample_balanced_trials(y: pd.Series, per_class: int, rng: np.random.Generator) -> List[int]:
		idx = []
		for c in sorted(y.unique()):
			trials_c = y[y == c].index.to_numpy()
			n_take = min(per_class, len(trials_c))
			idx.extend(rng.choice(trials_c, size=n_take, replace=False))
		return sorted(idx)

	def shuffle_within_stimulus_subset(X_full: pd.DataFrame, y_full: pd.Series, subset_trials: List[int], rng: np.random.Generator) -> np.ndarray:
		"""Return shuffled X for the given subset, using per-cell, per-stimulus permutations defined over the FULL set of trials for that stimulus.

		The returned array has shape (len(subset_trials), n_cells) aligned to X_full.columns.
		"""
		# Map trial_idx -> class and position within its class' global ordering
		classes = sorted(y_full.unique())
		# Build per-class global ordering and position map
		pos_map = {}
		order_map = {}
		for c in classes:
			mask_c = (y_full.values == c)
			trials_c = y_full.index.to_numpy()[mask_c]
			order_map[c] = trials_c
			pos_map[c] = {t: i for i, t in enumerate(trials_c)}
		n_cells = X_full.shape[1]
		# Prepare output
		X_shuff = np.empty((len(subset_trials), n_cells), dtype=float)
		# For efficiency, prefetch subset positions within each class
		subset_by_class = {c: [] for c in classes}
		for t in subset_trials:
			c = int(y_full.loc[t])
			subset_by_class[c].append(t)
		# For each cell, generate independent permutation for each class and fill values
		A_full = X_full.to_numpy(dtype=float)
		# Build a quick index: subset row index in output for each trial t
		row_index_of_t = {t: i for i, t in enumerate(subset_trials)}
		for j in range(n_cells):
			for c in classes:
				trials_c = order_map[c]
				if len(trials_c) == 0:
					continue
				vals_c = A_full[:, j][y_full.values == c]
				perm = rng.permutation(len(trials_c))
				vals_perm = vals_c[perm]
				# place into subset rows for trials in this class
				for t in subset_by_class[c]:
					pos = pos_map[c][t]
					X_shuff[row_index_of_t[t], j] = vals_perm[pos]
		return X_shuff

	def averaged_shuffled_boundary(X_full: pd.DataFrame, y_full: pd.Series, per_class: int, repeats: int, rng: np.random.Generator):
		"""Compute average decision boundary over `repeats` shuffled datasets in the real-data PLS space.

		Steps per time bin:
		- Draw a balanced trial subset for REAL data; fit scaler+PLS on real X to define the 2D space and fit green boundary.
		- For each repeat: draw a new balanced subset, shuffle per cell within stimulus based on full set, transform with real scaler+PLS, fit boundary; average normalized (w,b).
		"""
		# Real subset for PLS space
		subset_real = subsample_balanced_trials(y_full, per_class=per_class, rng=rng)
		X_real = X_full.loc[subset_real]
		y_real = y_full.loc[subset_real]
		pls, Z_real, y_enc_real, scaler = fit_pls_and_project(X_real, y_real, n_components=2)
		w_real, b_real = fit_linear_boundary(Z_real, y_enc_real)
		# Normalize for consistent sign (optional): enforce w_real[1] >= 0 to avoid flips when averaging
		if w_real[1] < 0:
			w_real = -w_real
			b_real = -b_real

		# Shuffled repeated boundaries in same Z space
		W_list = []
		B_list = []
		for r in range(repeats):
			subset = subsample_balanced_trials(y_full, per_class=per_class, rng=rng)
			X_shuff = shuffle_within_stimulus_subset(X_full, y_full, subset_trials=subset, rng=rng)
			Xs = scaler.transform(X_shuff)  # use real-data scaler
			Zs = pls.transform(Xs)         # project into real-data PLS space
			y_enc = np.where(y_full.loc[subset].values.astype(int) > 0, 1.0, -1.0)
			w, b = fit_linear_boundary(Zs, y_enc)
			# Normalize (w,b) to make averaging meaningful: unit-norm w and sign alignment
			norm = np.linalg.norm(w) + 1e-12
			w = w / norm
			b = b / norm
			if w[1] < 0:
				w = -w
				b = -b
			W_list.append(w)
			B_list.append(b)
		W_avg = np.mean(W_list, axis=0)
		B_avg = float(np.mean(B_list))
		return (pls, scaler, Z_real, y_enc_real, w_real, b_real, W_avg, B_avg)

	rng = np.random.default_rng(123)
	per_class = 220
	repeats = 100

	for i, t in enumerate(times):
		ax = axes[i]
		k = choose_bin_for_time_sec(t)
		print(f"LOG[L347]: Time {t:.3f}s -> sample_idx={k} (bin center={(k+0.5)*BIN_SEC:.3f}s)")
		X_full, y_full = build_full(mouse_id, k)
		print(f"LOG[L347]: Loaded full matrix: X={X_full.shape}, class counts={y_full.value_counts().to_dict()}")
		pls, scaler, Z_real, y_enc_real, w_real, b_real, W_avg, B_avg = averaged_shuffled_boundary(
			X_full, y_full, per_class=per_class, repeats=repeats, rng=rng
		)
		# Scatter real subset in PLS space
		m1 = y_enc_real > 0
		m2 = ~m1
		ax.scatter(Z_real[m1, 0], Z_real[m1, 1], c="#1f77b4", marker="x", label="stim + (real)", alpha=0.7)
		ax.scatter(Z_real[m2, 0], Z_real[m2, 1], c="#d62728", marker="x", label="stim - (real)", alpha=0.7)
		# Real boundary
		xmin, xmax = np.percentile(Z_real[:, 0], [1, 99])
		project_boundary_line(ax, w_real, b_real, color="green", label="real boundary", xmin=xmin, xmax=xmax)
		# Averaged shuffled boundary
		project_boundary_line(ax, W_avg, B_avg, color="orange", label="trial-shuffled (avg)", xmin=xmin, xmax=xmax, linestyle="--")
		# Diagonal discrimination from real
		thr = diagonal_discrimination_threshold(Z_real, y_enc_real)
		ax.axvline(thr, color="black", linestyle=":", label="diagonal discr.")
		ax.set_title(f"{mouse_id} @ {t:.2f}s (avg over {repeats} shuffles)")
		ax.set_xlabel("PLS comp 1")
		if i == 0:
			ax.set_ylabel("PLS comp 2")
		ax.grid(True, alpha=0.2)
		if i == 0:
			ax.legend(loc="best", fontsize=9)

	fig.suptitle("Mouse L347: PLS-DA with stimulus-conditional trial-shuffled boundary (avg of 100)")
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	out_dir2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results"))
	os.makedirs(out_dir2, exist_ok=True)
	out_path2 = os.path.join(out_dir2, "plsda_mouseL347_shuffled_avg.png")
	fig.savefig(out_path2, dpi=200)
	plt.close(fig)
	print("Saved Mouse_L347 shuffled-avg figure to:", out_path2)


if __name__ == "__main__":
	main()

