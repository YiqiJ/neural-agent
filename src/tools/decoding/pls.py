import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '../../..')

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Optional, Tuple, List
from pathlib import Path

from src.tools.normalize.normalize import standardize_data, minmaxscale_data

# --------- PLS MCP tool ---------
def pls_tool(dataset_root, subject_id, session_date, behavior_keys, region_keys=None, window_name=None, **kwargs) -> str:
    """PLS tool for neural-behavioral data.
    Args:
        dataset_root (str or Path): Root directory of the dataset.
        subject_id (str): Subject identifier.
        session_date (str): Session date identifier.
        window_name (str): Name of the trial window to use (stored under <dataset_root>/Subject-<subject_id>/Session-<session_date>/by_trials/). If None, use full session.
        behavior_key (str): Key for the behavioral variable in the behaviors array.
        n_components (int): Number of PLS components.
        n_splits (int): Number of cross-validation splits.
        seed (Optional[int]): Random state for reproducibility.
        X_normalize (Optional[str]): Normalization method for neural data ('zscore' or 'minmax').
        Y_normalize (Optional[str]): Normalization method for behavioral data ('zscore' or 'minmax').
    Returns:
        log: log summary of the operation including R^2 score.
    """
    from src.tools.dataset.dataset_api import DatasetAPI

    log = f"""
PLS tool for neural-behavioral data
Dataset root: {dataset_root}, Subject: {subject_id}, Session: {session_date}, Behavior key: {behavior_keys}\n"""

    api = DatasetAPI(dataset_root)
    session_ref = api.get_session(subject_id, session_date)

    # Load neural and behavioral data
    window_manifest_path = f"{session_ref.session_dir()}/by_trial/windows_{window_name}.json" if window_name else None
    if window_name and window_manifest_path and Path(window_manifest_path).exists():
        manifest = api.read_windows_manifest(window_manifest_path)
        windows = manifest.windows
        log += f"Using trial window: {window_name} with {len(windows)} trials.\n"
        neural_data = api.load_neural_trials(manifest, region_keys=region_keys)
        behavior_data = api.load_behavior_trials(manifest, behavior_keys)
    else:
        raise ValueError(f"Window name {window_name} not found or no window specified.")
        # windows = None
        # log += "No trial window specified or found; using full session data.\n"
        # neural_data = [api.load_neural_data(session_ref)]
        # behavior_data = [api.load_behavior_data(session_ref, behavior_keys)]

    # Perform PLS regression
    log += f"Running PLS regression with model parameters: {kwargs}\n"
    pls_model, r2_score_train, r2_score_test = pls_regression(neural_data, behavior_data, **kwargs)

    log += f"PLS regression completed with R^2 score (train): {r2_score_train}, R^2 score (test): {r2_score_test}\n"
    return log

# --------- helpers for PLS regression with cross-validation ---------
def pls_regression(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 2,
    n_splits: int = 5,
    normalize_X: Optional[str] = 'zscore',  # or 'minmax'
    normalize_Y: Optional[str] = 'zscore',   # or 'minmax'
    **kwargs
) -> Tuple[PLSRegression, float]:
    """
    Perform PLS regression with cross-validation.

    Args:
        
    Returns:
        Tuple[PLSRegression, float]: Fitted PLS model and average R^2 score across folds.
    """
    data_seed  = kwargs['data_seed']
    model_seed = kwargs['model_seed']
    # train-test split
    num_trials = len(X)
    val_size  = kwargs.get('val_size', 0.1)
    test_size = kwargs.get('test_size', 0.3)
    train_trial_idx, val_trial_idx, test_trials_idx = train_val_test_split(num_trials, val_size=val_size, test_size=test_size, seed=data_seed)
    train_trial_data = [X[i] for i in train_trial_idx]
    valid_trial_data = [X[i] for i in val_trial_idx] if len(val_trial_idx) != 0 else []
    test_trial_data  = [X[i] for i in test_trials_idx]
    train_behav_data = [Y[i] for i in train_trial_idx]
    valid_behav_data = [Y[i] for i in val_trial_idx] if len(val_trial_idx) != 0 else []
    test_behav_data  = [Y[i] for i in test_trials_idx]

    # normalize data
    train_trial_data, valid_trial_data, test_trial_data = normalize(normalize_X, train_trial_data, valid_trial_data, test_trial_data)
    train_behav_data, valid_behav_data, test_behav_data = normalize(normalize_Y, train_behav_data, valid_behav_data, test_behav_data)

    X_train = np.vstack(train_trial_data)
    Y_train = np.vstack(train_behav_data)
    X_test  = np.vstack(test_trial_data)
    Y_test  = np.vstack(test_behav_data)
    

    pls = PLSRegression(n_components=n_components)
    X_train = np.vstack(X_train)
    Y_train = np.vstack(Y_train)
    X_test  = np.vstack(X_test)
    Y_test  = np.vstack(Y_test)
    pls.fit(X_train, Y_train)

    Z_train = pls.transform(X_train)
    Z_test  = pls.transform(X_test)
    from sklearn import linear_model
    from sklearn.model_selection import RepeatedKFold, GridSearchCV
    lasso = linear_model.Lasso(alpha=0.1)
    params_grid = kwargs['params_grid']
    scoring     = kwargs['scoring']
    n_split     = kwargs['n_split']
    n_repeats   = kwargs['n_repeats']
    cv = RepeatedKFold(n_splits=n_split, n_repeats=n_repeats, random_state=model_seed)
    search = GridSearchCV(lasso, params_grid, scoring=scoring, cv=cv, n_jobs=1)
    decoder = search.fit(Z_train, Y_train)
    Y_pred_train = decoder.predict(Z_train)
    Y_pred_test  = decoder.predict(Z_test)
    # Y_pred_train = pls.predict(X_train)
    # Y_pred_test  = pls.predict(X_test)

    r2_train = r2_score(Y_train, Y_pred_train)
    r2_test  = r2_score(Y_test, Y_pred_test)

    return pls, r2_train, r2_test

def train_val_test_split(num_trials, val_size, test_size, seed):
    """
    Return train, test, validation indices split on [seed]
    """
    from sklearn.model_selection import train_test_split
    np.random.seed(seed)
    # get the train-test split based on trials
    trials = np.arange(num_trials)
    train_trial_idx, test_trial_idx = train_test_split(trials, test_size = test_size)
    if val_size != 0:
        train_trial_idx, val_trial_idx  = train_test_split(train_trial_idx, test_size = val_size)
    else:
        val_trial_idx = []

    return train_trial_idx, val_trial_idx, test_trial_idx

def normalize(type, train_trials, val_trials, test_trials):
    if type is not None:
        if type == 'zscore':
            train_trials, _mean, _std = standardize_data(train_trials)
            if len(val_trials) != 0:
                val_trials, _, _ = standardize_data(val_trials, _mean=_mean, _std=_std)        
            test_trials, _, _ = standardize_data(test_trials, _mean=_mean, _std=_std)
        elif type == 'minmax':
            train_trials, _max, _min = minmaxscale_data(train_trials)
            if len(val_trials) != 0:
                val_trials, _, _  = minmaxscale_data(val_trials, _max=_max, _min=_min)
            test_trials, _, _  = minmaxscale_data(test_trials, _max=_max, _min=_min)

    return train_trials, val_trials, test_trials