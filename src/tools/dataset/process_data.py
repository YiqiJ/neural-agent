import numpy as np
from dataset_api import DatasetAPI
from typing import List, Dict, Optional

def define_trial_date(dataset_root, subject_id, session_date, align_mode, **kwargs) -> Dict[str, int]:
    """Define trial windows for a given session and save a manifest.
    
    Args:
        dataset_root (str or Path): Root directory of the dataset.
        subject_id (str): Subject identifier.
        session_date (str): Session date identifier.
    Returns:
        log: log summary of the operation including number of trials and manifest path.
    """
    drop_oob = kwargs.get('drop_oob', False)
    min_trial_len = kwargs.get('min_trial_len', None)
    max_trial_len = kwargs.get('max_trial_len', None)

    log = f"""
Partition session into trials and save a manifest
Dataset root: {dataset_root}, Subject: {subject_id}, Session: {session_date}\n
"""

    api = DatasetAPI(dataset_root)
    session_ref = api.get_session(subject_id, session_date)

    if align_mode == 'peri_event':
        windows = api.compute_windows_event_window(
            session_ref,
            align_event=kwargs['align_event'],
            pre=kwargs['pre'],
            post=kwargs['post'],
            drop_oob=drop_oob,
            min_trial_len=min_trial_len,
            max_trial_len=max_trial_len,
        )
        log += f"Aligned to event '{kwargs['align_event']}' with pre={kwargs['pre']} and post={kwargs['post']} frames.\n"
    elif align_mode == 'event_range':
        windows = api.compute_windows_event_range(
            session_ref,
            start_event=kwargs['start_event'],
            end_event=kwargs['end_event'],
            drop_oob=drop_oob,
            min_trial_len=min_trial_len,
            max_trial_len=max_trial_len,
        )
        log += f"Between events '{kwargs['start_event']}' and '{kwargs['end_event']}'.\n"
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")
    log += f"Identified {len(windows)} trials.\n"

    manifest_path = api.write_windows_manifest(session_ref, windows, kwargs.get('out_dir', None))
    log += f"Saved manifest to: {manifest_path}\n"
    return log