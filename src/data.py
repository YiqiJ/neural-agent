from pathlib import Path

import numpy as np
import pandas as pd


def read_data(data_path: Path) -> pd.DataFrame:
	"""Read the parquet file with minimal overhead.

	Returns a DataFrame with columns:
	- mouse_id (object)
	- sample_idx (int32)
	- cell_idx (int32)
	- amplitude (float32)
	- trial_idx (int32)
	- behavior (int16, values {+30, -30})
	"""

	df = pd.read_parquet(data_path)
    # Basic dtypes
	df["mouse_id"] = df["mouse_id"].astype("category")
	for c in ["sample_idx", "cell_idx", "trial_idx", "behavior"]:
		if c in df:
			df[c] = pd.to_numeric(df[c], downcast="integer")
	if "amplitude" in df:
		df["amplitude"] = pd.to_numeric(df["amplitude"], downcast="float")
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
		df.groupby("mouse_id", observed=True)["cell_idx"].nunique().rename("n_cells").reset_index()
	)

	# trials per mouse
	trials_per_mouse = (
		df.groupby("mouse_id", observed=True)["trial_idx"].nunique().rename("n_trials").reset_index()
	)

	# behavior types
	behav_types = (
		df.groupby("mouse_id", observed=True)["behavior"]
		.apply(lambda s: ",".join(map(str, sorted(pd.unique(s)))))
		.rename("behavior_types")
		.reset_index()
	)

	# trial lengths: number of samples per (mouse, trial_idx)
	trial_lengths = (
		df.groupby(["mouse_id", "trial_idx"], observed=True)['sample_idx']
		.nunique()
		.rename("samples_per_trial")
		.reset_index()
	)
	tl_stats = (
		trial_lengths.groupby("mouse_id", observed=True)["samples_per_trial"]
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