#!/usr/bin/env python3
"""
Single mouse analysis of noise correlations for similarly vs differently tuned cell pairs.
Run this script for one mouse at a time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_mouse(mouse_id):
    """Analyze a single mouse."""
    print(f"Analyzing {mouse_id}...")
    
    # CHECKPOINT 1: Load data
    print("  CHECKPOINT 1: Loading data...")
    df = pd.read_parquet('data/coding_fidelity_bounds.dataset.parquet')
    print("  ✓ Data loaded successfully")
    
    mouse_df = df[df['mouse_id'] == mouse_id].copy()
    print("  ✓ Mouse data filtered")
    
    print(f"  Cells: {mouse_df['cell_idx'].nunique()}")
    print(f"  Trials: {mouse_df['trial_idx'].nunique()}")
    print(f"  Stimulus conditions: {sorted(mouse_df['behavior'].unique())}")
    
    # CHECKPOINT 2: Calculate mean responses
    print("  CHECKPOINT 2: Calculating mean responses...")
    mean_responses = {}
    for behavior in [-30, 30]:
        print(f"    Processing stimulus {behavior}°...")
        behavior_trials = mouse_df[mouse_df['behavior'] == behavior]
        mean_resp = behavior_trials.groupby('cell_idx')['amplitude'].mean()
        mean_responses[behavior] = mean_resp
        print(f"    ✓ Stimulus {behavior}°: {len(behavior_trials['trial_idx'].unique())} trials")
    print("  ✓ Mean responses calculated")
    
    # CHECKPOINT 3: Identify top cells
    print("  CHECKPOINT 3: Identifying top 10% most active cells...")
    mean_across_stimuli = (mean_responses[-30] + mean_responses[30]) / 2
    n_cells = len(mean_across_stimuli)
    n_top = int(n_cells * 0.1)
    top_cells = mean_across_stimuli.nlargest(n_top).index.values
    print(f"  ✓ Selected top {n_top} cells out of {n_cells} total")
    
    # CHECKPOINT 4: Calculate tuning correlations
    print("  CHECKPOINT 4: Calculating tuning correlations...")
    tuning_corrs = []
    cell_pairs = []
    
    total_pairs = len(top_cells) * (len(top_cells) - 1) // 2
    print(f"    Computing {total_pairs} cell pairs...")
    
    pair_count = 0
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
            
            pair_count += 1
            if pair_count % 1000 == 0:
                print(f"    Processed {pair_count}/{total_pairs} pairs...")
    
    tuning_corrs = np.array(tuning_corrs)
    print(f"  ✓ Calculated {len(tuning_corrs)} tuning correlations")
    
    # CHECKPOINT 5: Group pairs by tuning
    print("  CHECKPOINT 5: Grouping pairs by tuning...")
    pos_mask = tuning_corrs > 0
    neg_mask = tuning_corrs < 0
    
    pos_pairs = np.array(cell_pairs)[pos_mask]
    neg_pairs = np.array(cell_pairs)[neg_mask]
    
    print(f"  ✓ Positively tuned pairs: {len(pos_pairs)}")
    print(f"  ✓ Negatively tuned pairs: {len(neg_pairs)}")
    
    # CHECKPOINT 6: Calculate noise correlations
    print("  CHECKPOINT 6: Calculating noise correlations...")
    
    # Get trial data for top cells
    print("    Preparing trial data...")
    trial_data = {}
    for behavior in [-30, 30]:
        print(f"      Processing stimulus {behavior}° trial data...")
        behavior_trials = mouse_df[mouse_df['behavior'] == behavior]
        # Average across time samples for each trial
        trial_means = behavior_trials.groupby(['trial_idx', 'cell_idx'])['amplitude'].mean().unstack()
        trial_data[behavior] = trial_means[top_cells]  # Only top cells
        print(f"      ✓ Stimulus {behavior}°: {trial_means.shape[0]} trials")
    print("    ✓ Trial data prepared")
    
    # Calculate mean responses for noise calculation
    print("    Calculating mean responses for noise...")
    mean_resp_neg30 = trial_data[-30].mean()
    mean_resp_30 = trial_data[30].mean()
    print("    ✓ Mean responses calculated")
    
    # Calculate noise residuals
    print("    Calculating noise residuals...")
    noise_neg30 = trial_data[-30] - mean_resp_neg30
    noise_30 = trial_data[30] - mean_resp_30
    print("    ✓ Noise residuals calculated")
    
    # Combine noise across both conditions
    print("    Combining noise across conditions...")
    noise_combined = pd.concat([noise_neg30, noise_30])
    print(f"    ✓ Combined noise data: {noise_combined.shape}")
    
    # Calculate noise correlations for positively tuned pairs
    print(f"    Calculating noise correlations for {len(pos_pairs)} positively tuned pairs...")
    pos_noise_corrs = []
    for idx, (i, j) in enumerate(pos_pairs):
        corr, _ = pearsonr(noise_combined.iloc[:, i], noise_combined.iloc[:, j])
        pos_noise_corrs.append(corr)
        if (idx + 1) % 1000 == 0:
            print(f"      Processed {idx + 1}/{len(pos_pairs)} pos pairs...")
    print("    ✓ Positively tuned noise correlations calculated")
    
    # Calculate noise correlations for negatively tuned pairs
    print(f"    Calculating noise correlations for {len(neg_pairs)} negatively tuned pairs...")
    neg_noise_corrs = []
    for idx, (i, j) in enumerate(neg_pairs):
        corr, _ = pearsonr(noise_combined.iloc[:, i], noise_combined.iloc[:, j])
        neg_noise_corrs.append(corr)
        if (idx + 1) % 1000 == 0:
            print(f"      Processed {idx + 1}/{len(neg_pairs)} neg pairs...")
    print("    ✓ Negatively tuned noise correlations calculated")
    
    pos_noise_corrs = np.array(pos_noise_corrs)
    neg_noise_corrs = np.array(neg_noise_corrs)
    
    # CHECKPOINT 7: Clean data and statistical analysis
    print("  CHECKPOINT 7: Cleaning data and performing statistical analysis...")
    pos_clean = pos_noise_corrs[~np.isnan(pos_noise_corrs)]
    neg_clean = neg_noise_corrs[~np.isnan(neg_noise_corrs)]
    print(f"    Clean pos pairs: {len(pos_clean)}")
    print(f"    Clean neg pairs: {len(neg_clean)}")
    
    # Statistical test
    print("    Performing Kolmogorov-Smirnov test...")
    ks_stat, p_value = ks_2samp(pos_clean, neg_clean)
    print("    ✓ Statistical test completed")
    
    print(f"\n  Statistical Results:")
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Positively tuned: mean={np.mean(pos_clean):.4f}, std={np.std(pos_clean):.4f}")
    print(f"  Negatively tuned: mean={np.mean(neg_clean):.4f}, std={np.std(neg_clean):.4f}")
    
    # CHECKPOINT 8: Create visualization
    print("  CHECKPOINT 8: Creating histogram...")
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
    filename = f'tuning_correlation_histogram_{mouse_id}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    ✓ Plot saved as: {filename}")
    plt.show()
    print("  ✓ Histogram created and displayed")
    
    # CHECKPOINT 9: Return results
    print("  CHECKPOINT 9: Compiling results...")
    results = {
        'mouse_id': mouse_id,
        'n_cells': n_cells,
        'n_top_cells': n_top,
        'n_pos_pairs': len(pos_clean),
        'n_neg_pairs': len(neg_clean),
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'pos_mean': np.mean(pos_clean),
        'pos_std': np.std(pos_clean),
        'neg_mean': np.mean(neg_clean),
        'neg_std': np.std(neg_clean)
    }
    print("  ✓ Results compiled")
    
    return results

def main():
    """Main function - specify which mouse to analyze."""
    import sys
    
    if len(sys.argv) > 1:
        mouse_id = sys.argv[1]
    else:
        # Default to first mouse
        df = pd.read_parquet('../data/coding_fidelity_bounds.dataset.parquet')
        mouse_id = df['mouse_id'].unique()[0]
    
    print("="*70)
    print(f"ANALYZING MOUSE: {mouse_id}")
    print("="*70)
    
    result = analyze_mouse(mouse_id)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    return result

if __name__ == "__main__":
    main()
