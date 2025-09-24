#!/usr/bin/env python3
"""
Create combined histogram showing noise correlations for all 5 mice.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr

def analyze_mouse_for_combined(mouse_id):
    """Analyze a single mouse and return noise correlations."""
    print(f"Analyzing {mouse_id}...")
    
    # Load data
    df = pd.read_parquet('data/oleg_data.parquet')
    mouse_df = df[df['mouse_id'] == mouse_id].copy()
    
    # Calculate mean responses
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
    
    # Calculate tuning correlations
    tuning_corrs = []
    cell_pairs = []
    
    for i in range(len(top_cells)):
        for j in range(i+1, len(top_cells)):
            cell1, cell2 = top_cells[i], top_cells[j]
            resp1 = [mean_responses[-30][cell1], mean_responses[30][cell1]]
            resp2 = [mean_responses[-30][cell2], mean_responses[30][cell2]]
            corr, _ = pearsonr(resp1, resp2)
            tuning_corrs.append(corr)
            cell_pairs.append((i, j))
    
    tuning_corrs = np.array(tuning_corrs)
    
    # Group pairs
    pos_mask = tuning_corrs > 0
    neg_mask = tuning_corrs < 0
    
    pos_pairs = np.array(cell_pairs)[pos_mask]
    neg_pairs = np.array(cell_pairs)[neg_mask]
    
    # Calculate noise correlations
    trial_data = {}
    for behavior in [-30, 30]:
        behavior_trials = mouse_df[mouse_df['behavior'] == behavior]
        trial_means = behavior_trials.groupby(['trial_idx', 'cell_idx'])['amplitude'].mean().unstack()
        trial_data[behavior] = trial_means[top_cells]
    
    # Calculate noise residuals
    mean_resp_neg30 = trial_data[-30].mean()
    mean_resp_30 = trial_data[30].mean()
    noise_neg30 = trial_data[-30] - mean_resp_neg30
    noise_30 = trial_data[30] - mean_resp_30
    noise_combined = pd.concat([noise_neg30, noise_30])
    
    # Calculate noise correlations for pos pairs
    pos_noise_corrs = []
    for i, j in pos_pairs:
        corr, _ = pearsonr(noise_combined.iloc[:, i], noise_combined.iloc[:, j])
        pos_noise_corrs.append(corr)
    
    # Calculate noise correlations for neg pairs
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
    
    print(f"  ✓ {mouse_id}: {len(pos_clean)} pos pairs, {len(neg_clean)} neg pairs, P = {p_value:.2e}")
    
    return {
        'mouse_id': mouse_id,
        'pos_corrs': pos_clean,
        'neg_corrs': neg_clean,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'pos_mean': np.mean(pos_clean),
        'neg_mean': np.mean(neg_clean)
    }

def create_combined_histogram(all_results):
    """Create combined histogram for all mice."""
    print("Creating combined histogram...")
    
    # Combine all data
    all_pos_corrs = np.concatenate([result['pos_corrs'] for result in all_results])
    all_neg_corrs = np.concatenate([result['neg_corrs'] for result in all_results])
    
    print(f"Combined statistics:")
    print(f"  Total pos pairs: {len(all_pos_corrs):,}")
    print(f"  Total neg pairs: {len(all_neg_corrs):,}")
    print(f"  Pos mean: {np.mean(all_pos_corrs):.4f} ± {np.std(all_pos_corrs):.4f}")
    print(f"  Neg mean: {np.mean(all_neg_corrs):.4f} ± {np.std(all_neg_corrs):.4f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Individual mouse histograms
    colors_pos = ['red', 'darkred', 'crimson', 'firebrick', 'indianred']
    colors_neg = ['blue', 'darkblue', 'navy', 'steelblue', 'lightblue']
    
    for i, result in enumerate(all_results):
        ax = axes[i]
        
        bins = np.linspace(-0.5, 0.5, 50)
        ax.hist(result['pos_corrs'], bins=bins, alpha=0.6, 
                label=f'Pos tuned (n={len(result["pos_corrs"])})', 
                color=colors_pos[i], density=True, edgecolor='black', linewidth=0.5)
        ax.hist(result['neg_corrs'], bins=bins, alpha=0.6, 
                label=f'Neg tuned (n={len(result["neg_corrs"])})', 
                color=colors_neg[i], density=True, edgecolor='black', linewidth=0.5)
        
        # Add mean lines
        ax.axvline(result['pos_mean'], color=colors_pos[i], linestyle='--', linewidth=2)
        ax.axvline(result['neg_mean'], color=colors_neg[i], linestyle='--', linewidth=2)
        
        ax.set_xlabel('Noise Correlation Coefficient')
        ax.set_ylabel('Density')
        ax.set_title(f'{result["mouse_id"]}\nP = {result["p_value"]:.2e}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 0.5)
    
    # Combined histogram
    ax_combined = axes[5]
    bins = np.linspace(-0.5, 0.5, 50)
    ax_combined.hist(all_pos_corrs, bins=bins, alpha=0.6, 
                     label=f'Positively tuned pairs (n={len(all_pos_corrs):,})', 
                     color='red', edgecolor='black', linewidth=0.5)
    ax_combined.hist(all_neg_corrs, bins=bins, alpha=0.6, 
                     label=f'Negatively tuned pairs (n={len(all_neg_corrs):,})', 
                     color='blue', edgecolor='black', linewidth=0.5)
    
    # Add mean lines
    ax_combined.axvline(np.mean(all_pos_corrs), color='red', linestyle='--', linewidth=3, 
                        label=f"Pos mean: {np.mean(all_pos_corrs):.3f}")
    ax_combined.axvline(np.mean(all_neg_corrs), color='blue', linestyle='--', linewidth=3, 
                        label=f"Neg mean: {np.mean(all_neg_corrs):.3f}")
    
    ax_combined.set_xlabel('Noise Correlation Coefficient', fontsize=12)
    ax_combined.set_ylabel('Number of Cell Pairs', fontsize=12)
    ax_combined.set_title('Combined Results - All 5 Mice', 
                         fontsize=14, fontweight='bold')
    ax_combined.legend(fontsize=10)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_xlim(-0.5, 0.5)
    
    # Add statistical information (without P-value)
    stats_text = f'Combined Statistics:\n'
    stats_text += f'Total pairs: {len(all_pos_corrs) + len(all_neg_corrs):,}\n'
    stats_text += f'Pos mean: {np.mean(all_pos_corrs):.4f} ± {np.std(all_pos_corrs):.4f}\n'
    stats_text += f'Neg mean: {np.mean(all_neg_corrs):.4f} ± {np.std(all_neg_corrs):.4f}'
    
    ax_combined.text(0.02, 0.98, stats_text, transform=ax_combined.transAxes, 
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = 'combined_noise_correlation_histogram_all_mice.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved combined histogram: {filename}")
    
    plt.show()
    
    return {
        'all_pos_corrs': all_pos_corrs,
        'all_neg_corrs': all_neg_corrs,
        'pos_mean': np.mean(all_pos_corrs),
        'neg_mean': np.mean(all_neg_corrs)
    }

def main():
    """Main function to create combined histogram."""
    print("="*70)
    print("CREATING COMBINED HISTOGRAM FOR ALL 5 MICE")
    print("="*70)
    
    mouse_ids = ['Mouse_L347', 'Mouse_L354', 'Mouse_L355', 'Mouse_L362', 'Mouse_L363']
    
    # Analyze all mice
    all_results = []
    for mouse_id in mouse_ids:
        result = analyze_mouse_for_combined(mouse_id)
        all_results.append(result)
    
    # Create combined histogram
    combined_stats = create_combined_histogram(all_results)
    
    print(f"\n{'='*70}")
    print("COMBINED ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Total positively tuned pairs: {len(combined_stats['all_pos_corrs']):,}")
    print(f"Total negatively tuned pairs: {len(combined_stats['all_neg_corrs']):,}")
    print(f"Pos mean: {combined_stats['pos_mean']:.4f}")
    print(f"Neg mean: {combined_stats['neg_mean']:.4f}")

if __name__ == "__main__":
    main()
