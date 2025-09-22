#!/usr/bin/env python3
"""
Quick analysis of noise correlations for similarly vs differently tuned cell pairs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_parquet('data/coding_fidelity_bounds.dataset.parquet')
    
    # Focus on one mouse for speed
    mouse_id = 'Mouse_L347'
    mouse_df = df[df['mouse_id'] == mouse_id].copy()
    
    print(f"Analyzing {mouse_id}: {mouse_df['cell_idx'].nunique()} cells, {mouse_df['trial_idx'].nunique()} trials")
    
    # Calculate mean responses for each cell to each stimulus
    mean_responses = {}
    for behavior in [-30, 30]:
        behavior_trials = mouse_df[mouse_df['behavior'] == behavior]
        mean_resp = behavior_trials.groupby('cell_idx')['amplitude'].mean()
        mean_responses[behavior] = mean_resp
    
    # Get top 10% most active cells
    mean_across_stimuli = (mean_responses[-30] + mean_responses[30]) / 2
    n_cells = len(mean_across_stimuli)
    n_top = int(n_cells * 0.1)
    top_cells = mean_across_stimuli.nlargest(n_top).index.values
    
    print(f"Selected top {n_top} cells out of {n_cells}")
    
    # Calculate tuning correlations for top cells
    tuning_corrs = []
    cell_pairs = []
    
    for i in range(len(top_cells)):
        for j in range(i+1, len(top_cells)):
            cell1, cell2 = top_cells[i], top_cells[j]
            
            # Get mean responses for this pair
            resp1 = [mean_responses[-30][cell1], mean_responses[30][cell1]]
            resp2 = [mean_responses[-30][cell2], mean_responses[30][cell2]]
            
            # Calculate tuning correlation
            corr, _ = pearsonr(resp1, resp2)
            tuning_corrs.append(corr)
            cell_pairs.append((i, j))
    
    tuning_corrs = np.array(tuning_corrs)
    
    # Group pairs by tuning
    pos_mask = tuning_corrs > 0
    neg_mask = tuning_corrs < 0
    
    pos_pairs = np.array(cell_pairs)[pos_mask]
    neg_pairs = np.array(cell_pairs)[neg_mask]
    
    print(f"Positively tuned pairs: {len(pos_pairs)}")
    print(f"Negatively tuned pairs: {len(neg_pairs)}")
    
    # Calculate noise correlations for each group
    # First, get trial data for top cells
    trial_data = {}
    for behavior in [-30, 30]:
        behavior_trials = mouse_df[mouse_df['behavior'] == behavior]
        # Average across time samples for each trial
        trial_means = behavior_trials.groupby(['trial_idx', 'cell_idx'])['amplitude'].mean().unstack()
        trial_data[behavior] = trial_means[top_cells]  # Only top cells
    
    # Calculate mean responses for noise calculation
    mean_resp_neg30 = trial_data[-30].mean()
    mean_resp_30 = trial_data[30].mean()
    
    # Calculate noise residuals
    noise_neg30 = trial_data[-30] - mean_resp_neg30
    noise_30 = trial_data[30] - mean_resp_30
    
    # Combine noise across both conditions
    noise_combined = pd.concat([noise_neg30, noise_30])
    
    # Calculate noise correlations for positively tuned pairs
    pos_noise_corrs = []
    for i, j in pos_pairs:
        corr, _ = pearsonr(noise_combined.iloc[:, i], noise_combined.iloc[:, j])
        pos_noise_corrs.append(corr)
    
    # Calculate noise correlations for negatively tuned pairs
    neg_noise_corrs = []
    for i, j in neg_pairs:
        corr, _ = pearsonr(noise_combined.iloc[:, i], noise_combined.iloc[:, j])
        neg_noise_corrs.append(corr)
    
    pos_noise_corrs = np.array(pos_noise_corrs)
    neg_noise_corrs = np.array(neg_noise_corrs)
    
    # Remove NaN values
    pos_clean = pos_noise_corrs[~np.isnan(pos_noise_corrs)]
    neg_clean = neg_noise_corrs[~np.isnan(neg_noise_corrs)]
    
    # Statistical test
    ks_stat, p_value = ks_2samp(pos_clean, neg_clean)
    
    print(f"\nStatistical Results:")
    print(f"KS statistic: {ks_stat:.6f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Positively tuned: mean={np.mean(pos_clean):.4f}, std={np.std(pos_clean):.4f}")
    print(f"Negatively tuned: mean={np.mean(neg_clean):.4f}, std={np.std(neg_clean):.4f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(-0.5, 0.5, 50)
    plt.hist(pos_clean, bins=bins, alpha=0.6, label=f'Positively tuned pairs (n={len(pos_clean)})', 
             color='red', density=True, edgecolor='black')
    plt.hist(neg_clean, bins=bins, alpha=0.6, label=f'Negatively tuned pairs (n={len(neg_clean)})', 
             color='blue', density=True, edgecolor='black')
    
    # Add mean lines
    plt.axvline(np.mean(pos_clean), color='red', linestyle='--', linewidth=2, 
                label=f"Pos mean: {np.mean(pos_clean):.3f}")
    plt.axvline(np.mean(neg_clean), color='blue', linestyle='--', linewidth=2, 
                label=f"Neg mean: {np.mean(neg_clean):.3f}")
    
    plt.xlabel('Noise Correlation Coefficient')
    plt.ylabel('Density')
    plt.title(f'Noise Correlations: Similarly vs Differently Tuned Pairs\nMouse {mouse_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistical info
    p_val_str = f"P = {p_value:.2e}" if p_value < 0.001 else f"P = {p_value:.3f}"
    plt.text(0.02, 0.98, f'Kolmogorov-Smirnov test:\n{p_val_str}\nKS = {ks_stat:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'tuning_correlation_histogram_{mouse_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: tuning_correlation_histogram_{mouse_id}.png")

if __name__ == "__main__":
    main()
