#!/usr/bin/env python3
"""
Analysis of noise correlations for similarly vs differently tuned cell pairs.

This script implements the specific analysis described in the paper:
- Identifies top 10% most active cells
- Groups cell pairs by tuning similarity (positive vs negative correlation of mean responses)
- Compares noise correlation distributions between these groups
- Performs statistical testing (Kolmogorov-Smirnov test)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_neural_data(filepath):
    """Load neural activity data from parquet file."""
    print("Loading neural activity data...")
    df = pd.read_parquet(filepath)
    print(f"Data shape: {df.shape}")
    print(f"Unique behaviors: {sorted(df['behavior'].unique())}")
    return df

def preprocess_data_for_tuning(df, mouse_id):
    """
    Preprocess data to calculate mean responses and tuning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw neural activity data
    mouse_id : str
        Mouse ID to analyze
    
    Returns:
    --------
    dict : Preprocessed data
    """
    print(f"Preprocessing data for {mouse_id}...")
    
    # Filter for specific mouse
    mouse_df = df[df['mouse_id'] == mouse_id].copy()
    
    # Get basic info
    n_cells = mouse_df['cell_idx'].nunique()
    n_trials = mouse_df['trial_idx'].nunique()
    n_samples = mouse_df.groupby('trial_idx')['sample_idx'].nunique().iloc[0]
    behaviors = sorted(mouse_df['behavior'].unique())
    
    print(f"  Cells: {n_cells}, Trials: {n_trials}, Samples per trial: {n_samples}")
    print(f"  Stimulus conditions: {behaviors}")
    
    # Reshape data: (n_trials, n_cells, n_samples)
    pivot_data = mouse_df.pivot_table(
        index=['trial_idx', 'cell_idx'], 
        columns='sample_idx', 
        values='amplitude', 
        fill_value=0
    )
    
    # Get trial metadata
    trial_metadata = mouse_df.groupby('trial_idx')['behavior'].first()
    
    # Reshape to (n_trials, n_cells, n_samples)
    n_trials_actual = len(pivot_data.index.get_level_values('trial_idx').unique())
    neural_activity = np.zeros((n_trials_actual, n_cells, n_samples))
    
    for i, (trial_idx, cell_idx) in enumerate(pivot_data.index):
        trial_num = trial_idx
        cell_num = cell_idx
        if trial_num < n_trials_actual and cell_num < n_cells:
            neural_activity[trial_num, cell_num, :] = pivot_data.loc[(trial_idx, cell_idx)].values
    
    return {
        'neural_activity': neural_activity,  # (n_trials, n_cells, n_samples)
        'trial_metadata': trial_metadata,
        'n_cells': n_cells,
        'n_trials': n_trials_actual,
        'n_samples': n_samples,
        'behaviors': behaviors
    }

def calculate_mean_responses(neural_activity, trial_metadata, behaviors):
    """
    Calculate mean responses for each cell to each stimulus condition.
    
    Parameters:
    -----------
    neural_activity : np.ndarray
        Neural activity array of shape (n_trials, n_cells, n_samples)
    trial_metadata : pandas.Series
        Stimulus condition for each trial
    behaviors : list
        List of unique stimulus conditions
    
    Returns:
    --------
    dict : Mean responses for each stimulus condition
    """
    print("Calculating mean responses...")
    
    mean_responses = {}
    
    for behavior in behaviors:
        # Get trials for this stimulus condition
        trial_mask = trial_metadata == behavior
        trials_for_stimulus = neural_activity[trial_mask]  # (n_trials_for_stimulus, n_cells, n_samples)
        
        # Calculate mean across trials and time for each cell
        mean_response = np.mean(trials_for_stimulus, axis=(0, 2))  # (n_cells,)
        mean_responses[behavior] = mean_response
        
        print(f"  Stimulus {behavior}°: {np.sum(trial_mask)} trials, mean response: {np.mean(mean_response):.4f}")
    
    return mean_responses

def identify_top_active_cells(mean_responses, behaviors, top_percent=10):
    """
    Identify top most active cells based on mean response across both stimuli.
    
    Parameters:
    -----------
    mean_responses : dict
        Mean responses for each stimulus condition
    behaviors : list
        List of stimulus conditions
    top_percent : float
        Percentage of top cells to select (default 10%)
    
    Returns:
    --------
    np.ndarray : Indices of top active cells
    """
    print(f"Identifying top {top_percent}% most active cells...")
    
    # Calculate mean response across both stimuli for each cell
    mean_across_stimuli = np.mean([mean_responses[behavior] for behavior in behaviors], axis=0)
    
    # Get top cells
    n_cells = len(mean_across_stimuli)
    n_top_cells = int(n_cells * top_percent / 100)
    top_cell_indices = np.argsort(mean_across_stimuli)[-n_top_cells:]
    
    print(f"  Selected {n_top_cells} cells out of {n_cells} total")
    print(f"  Top cell mean response: {np.mean(mean_across_stimuli[top_cell_indices]):.4f}")
    
    return top_cell_indices

def calculate_tuning_correlations(mean_responses, behaviors, top_cell_indices):
    """
    Calculate tuning correlations between cell pairs.
    
    Parameters:
    -----------
    mean_responses : dict
        Mean responses for each stimulus condition
    behaviors : list
        List of stimulus conditions
    top_cell_indices : np.ndarray
        Indices of top active cells
    
    Returns:
    --------
    tuple : (tuning_correlations, cell_pairs)
    """
    print("Calculating tuning correlations...")
    
    # Get mean responses for top cells only
    mean_resp_1 = mean_responses[behaviors[0]][top_cell_indices]  # Response to first stimulus
    mean_resp_2 = mean_responses[behaviors[1]][top_cell_indices]  # Response to second stimulus
    
    n_cells = len(top_cell_indices)
    tuning_correlations = []
    cell_pairs = []
    
    # Calculate pairwise tuning correlations
    for i in range(n_cells):
        for j in range(i+1, n_cells):
            # Get mean responses for this cell pair
            resp1_pair = [mean_resp_1[i], mean_resp_1[j]]
            resp2_pair = [mean_resp_2[i], mean_resp_2[j]]
            
            # Calculate correlation between mean responses
            corr, _ = pearsonr(resp1_pair, resp2_pair)
            tuning_correlations.append(corr)
            cell_pairs.append((i, j))
    
    tuning_correlations = np.array(tuning_correlations)
    cell_pairs = np.array(cell_pairs)
    
    print(f"  Calculated {len(tuning_correlations)} cell pair correlations")
    print(f"  Mean tuning correlation: {np.mean(tuning_correlations):.4f}")
    
    return tuning_correlations, cell_pairs

def group_cell_pairs_by_tuning(tuning_correlations, cell_pairs):
    """
    Group cell pairs by tuning similarity.
    
    Parameters:
    -----------
    tuning_correlations : np.ndarray
        Tuning correlations for each cell pair
    cell_pairs : np.ndarray
        Cell pair indices
    
    Returns:
    --------
    tuple : (pos_tuned_pairs, neg_tuned_pairs, pos_indices, neg_indices)
    """
    print("Grouping cell pairs by tuning...")
    
    # Group pairs by positive vs negative tuning correlation
    pos_mask = tuning_correlations > 0
    neg_mask = tuning_correlations < 0
    
    pos_tuned_pairs = cell_pairs[pos_mask]
    neg_tuned_pairs = cell_pairs[neg_mask]
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]
    
    print(f"  Positively tuned pairs: {len(pos_tuned_pairs)}")
    print(f"  Negatively tuned pairs: {len(neg_tuned_pairs)}")
    
    return pos_tuned_pairs, neg_tuned_pairs, pos_indices, neg_indices

def calculate_noise_correlations_for_groups(neural_activity, trial_metadata, behaviors, 
                                          top_cell_indices, pos_indices, neg_indices):
    """
    Calculate noise correlations for positively and negatively tuned cell pairs.
    
    Parameters:
    -----------
    neural_activity : np.ndarray
        Neural activity array of shape (n_trials, n_cells, n_samples)
    trial_metadata : pandas.Series
        Stimulus condition for each trial
    behaviors : list
        List of stimulus conditions
    top_cell_indices : np.ndarray
        Indices of top active cells
    pos_indices : np.ndarray
        Indices of positively tuned pairs
    neg_indices : np.ndarray
        Indices of negatively tuned pairs
    
    Returns:
    --------
    tuple : (pos_noise_corrs, neg_noise_corrs)
    """
    print("Calculating noise correlations for grouped pairs...")
    
    # Calculate mean stimulus responses for top cells
    stimulus_means = {}
    for behavior in behaviors:
        trial_mask = trial_metadata == behavior
        trials_for_stimulus = neural_activity[trial_mask]
        mean_response = np.mean(trials_for_stimulus, axis=0)  # (n_cells, n_samples)
        stimulus_means[behavior] = mean_response[top_cell_indices]  # Only top cells
    
    # Calculate noise residuals for top cells
    n_trials, _, n_samples = neural_activity.shape
    n_top_cells = len(top_cell_indices)
    noise_residuals = np.zeros((n_trials, n_top_cells, n_samples))
    
    for trial_idx in range(n_trials):
        behavior = trial_metadata.iloc[trial_idx]
        mean_response = stimulus_means[behavior]  # (n_top_cells, n_samples)
        trial_activity = neural_activity[trial_idx, top_cell_indices, :]  # (n_top_cells, n_samples)
        noise_residuals[trial_idx] = trial_activity - mean_response
    
    # Average across time samples
    avg_noise_residuals = np.mean(noise_residuals, axis=2)  # (n_trials, n_top_cells)
    
    # Calculate noise correlations for positively tuned pairs
    pos_noise_corrs = []
    for idx in pos_indices:
        i, j = idx // (n_top_cells - 1), idx % (n_top_cells - 1)
        if j >= i:
            j += 1  # Skip diagonal
        corr, _ = pearsonr(avg_noise_residuals[:, i], avg_noise_residuals[:, j])
        pos_noise_corrs.append(corr)
    
    # Calculate noise correlations for negatively tuned pairs
    neg_noise_corrs = []
    for idx in neg_indices:
        i, j = idx // (n_top_cells - 1), idx % (n_top_cells - 1)
        if j >= i:
            j += 1  # Skip diagonal
        corr, _ = pearsonr(avg_noise_residuals[:, i], avg_noise_residuals[:, j])
        neg_noise_corrs.append(corr)
    
    pos_noise_corrs = np.array(pos_noise_corrs)
    neg_noise_corrs = np.array(neg_noise_corrs)
    
    print(f"  Positively tuned noise correlations: {len(pos_noise_corrs)} pairs")
    print(f"  Negatively tuned noise correlations: {len(neg_noise_corrs)} pairs")
    
    return pos_noise_corrs, neg_noise_corrs

def perform_statistical_test(pos_noise_corrs, neg_noise_corrs):
    """
    Perform Kolmogorov-Smirnov test to compare distributions.
    
    Parameters:
    -----------
    pos_noise_corrs : np.ndarray
        Noise correlations for positively tuned pairs
    neg_noise_corrs : np.ndarray
        Noise correlations for negatively tuned pairs
    
    Returns:
    --------
    dict : Statistical test results
    """
    print("Performing Kolmogorov-Smirnov test...")
    
    # Remove NaN values
    pos_clean = pos_noise_corrs[~np.isnan(pos_noise_corrs)]
    neg_clean = neg_noise_corrs[~np.isnan(neg_noise_corrs)]
    
    # Perform two-tailed KS test
    ks_statistic, p_value = ks_2samp(pos_clean, neg_clean)
    
    # Calculate additional statistics
    stats = {
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'n_pos_pairs': len(pos_clean),
        'n_neg_pairs': len(neg_clean),
        'pos_mean': np.mean(pos_clean),
        'pos_std': np.std(pos_clean),
        'neg_mean': np.mean(neg_clean),
        'neg_std': np.std(neg_clean),
        'pos_median': np.median(pos_clean),
        'neg_median': np.median(neg_clean)
    }
    
    print(f"  KS statistic: {ks_statistic:.6f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Positively tuned: mean={stats['pos_mean']:.4f}, std={stats['pos_std']:.4f}")
    print(f"  Negatively tuned: mean={stats['neg_mean']:.4f}, std={stats['neg_std']:.4f}")
    
    return stats

def create_histogram_comparison(pos_noise_corrs, neg_noise_corrs, stats, mouse_id, save_plot=True):
    """
    Create histogram comparison of noise correlations for differently tuned pairs.
    
    Parameters:
    -----------
    pos_noise_corrs : np.ndarray
        Noise correlations for positively tuned pairs
    neg_noise_corrs : np.ndarray
        Noise correlations for negatively tuned pairs
    stats : dict
        Statistical test results
    mouse_id : str
        Mouse ID for plot title
    save_plot : bool
        Whether to save the plot
    """
    print("Creating histogram comparison...")
    
    # Remove NaN values
    pos_clean = pos_noise_corrs[~np.isnan(pos_noise_corrs)]
    neg_clean = neg_noise_corrs[~np.isnan(neg_noise_corrs)]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histograms
    bins = np.linspace(-0.5, 0.5, 50)
    
    ax.hist(pos_clean, bins=bins, alpha=0.6, label=f'Positively tuned pairs (n={len(pos_clean)})', 
            color='red', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(neg_clean, bins=bins, alpha=0.6, label=f'Negatively tuned pairs (n={len(neg_clean)})', 
            color='blue', density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for means
    ax.axvline(stats['pos_mean'], color='red', linestyle='--', linewidth=2, 
               label=f"Pos mean: {stats['pos_mean']:.3f}")
    ax.axvline(stats['neg_mean'], color='blue', linestyle='--', linewidth=2, 
               label=f"Neg mean: {stats['neg_mean']:.3f}")
    
    # Formatting
    ax.set_xlabel('Noise Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Noise Correlations: Similarly vs Differently Tuned Pairs\nMouse {mouse_id}', 
                 fontsize=14, fontweight='bold')
    
    # Add statistical information
    p_val_str = f"P = {stats['p_value']:.2e}" if stats['p_value'] < 0.001 else f"P = {stats['p_value']:.3f}"
    ax.text(0.02, 0.98, f'Kolmogorov-Smirnov test:\n{p_val_str}\nKS = {stats["ks_statistic"]:.4f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 0.5)
    
    plt.tight_layout()
    
    if save_plot:
        filename = f'tuning_correlation_histogram_{mouse_id}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    plt.show()

def analyze_all_mice(df):
    """Analyze all mice and create combined results."""
    print("Analyzing all mice...")
    
    mouse_ids = df['mouse_id'].unique()
    all_results = {}
    
    for mouse_id in mouse_ids:
        print(f"\n{'='*50}")
        print(f"ANALYZING MOUSE: {mouse_id}")
        print(f"{'='*50}")
        
        # Preprocess data
        data = preprocess_data_for_tuning(df, mouse_id)
        
        # Calculate mean responses
        mean_responses = calculate_mean_responses(data['neural_activity'], 
                                                data['trial_metadata'], 
                                                data['behaviors'])
        
        # Identify top active cells
        top_cell_indices = identify_top_active_cells(mean_responses, data['behaviors'])
        
        # Calculate tuning correlations
        tuning_correlations, cell_pairs = calculate_tuning_correlations(mean_responses, 
                                                                       data['behaviors'], 
                                                                       top_cell_indices)
        
        # Group cell pairs
        pos_tuned_pairs, neg_tuned_pairs, pos_indices, neg_indices = group_cell_pairs_by_tuning(
            tuning_correlations, cell_pairs)
        
        # Calculate noise correlations
        pos_noise_corrs, neg_noise_corrs = calculate_noise_correlations_for_groups(
            data['neural_activity'], data['trial_metadata'], data['behaviors'],
            top_cell_indices, pos_indices, neg_indices)
        
        # Perform statistical test
        stats = perform_statistical_test(pos_noise_corrs, neg_noise_corrs)
        
        # Create histogram
        create_histogram_comparison(pos_noise_corrs, neg_noise_corrs, stats, mouse_id)
        
        # Store results
        all_results[mouse_id] = {
            'pos_noise_corrs': pos_noise_corrs,
            'neg_noise_corrs': neg_noise_corrs,
            'stats': stats,
            'n_top_cells': len(top_cell_indices),
            'n_pos_pairs': len(pos_tuned_pairs),
            'n_neg_pairs': len(neg_tuned_pairs)
        }
    
    return all_results

def main():
    """Main function to run the tuning correlation analysis."""
    print("="*70)
    print("NOISE CORRELATION ANALYSIS: SIMILARLY vs DIFFERENTLY TUNED PAIRS")
    print("="*70)
    
    # Load data
    df = load_neural_data('data/oleg_data.parquet')
    
    # Analyze all mice
    results = analyze_all_mice(df)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL MICE")
    print(f"{'='*70}")
    
    for mouse_id, result in results.items():
        stats = result['stats']
        print(f"\n{mouse_id}:")
        print(f"  Top cells: {result['n_top_cells']}")
        print(f"  Positively tuned pairs: {result['n_pos_pairs']}")
        print(f"  Negatively tuned pairs: {result['n_neg_pairs']}")
        print(f"  P-value: {stats['p_value']:.2e}")
        print(f"  Pos mean: {stats['pos_mean']:.4f} ± {stats['pos_std']:.4f}")
        print(f"  Neg mean: {stats['neg_mean']:.4f} ± {stats['neg_std']:.4f}")

if __name__ == "__main__":
    main()
