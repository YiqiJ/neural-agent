# dataset API
from pathlib import Path
from typing import List, Tuple, Union, Optional, Iterable, Iterator, Dict
from dataclasses import dataclass
import json 
import numpy as np
import zarr 
import pandas as pd

@dataclass(frozen=True)
class SessionRef:
    """Reference to a session in the dataset."""
    root: Path 
    subject_id: str
    session_date: str 

    def session_dir(self) -> Path:
        return self.root / Path(f"Subject_{self.subject_id}") / Path(f"Session_{self.session_date}")

@dataclass(frozen=True)
class Window:
    """A window for each trial"""
    start: int 
    end: int 
    @property
    def length(self) -> int: return self.end - self.start

@dataclass
class WindowsManifest:
    """Manifest for windows in a session."""
    session: SessionRef
    arrays: Dict[str, str] #{'traces': '../traces.zarr', 'behaviors': '../behaviors.zarr'}
    windows: List[Window]

class DatasetAPI:
    def __init__(self, dataset_root: Union[str, Path]):
        self.dataset_root = Path(dataset_root)

    # ------------------------- zarr helpers -------------------------
    def _open_dataset(self, store_path: Path, dataset_name: Optional[str] = None):
        """Open a Zarr dataset regardless of whether the path is a group store or a bare array store.

        If the path points to a DirectoryStore created via `root = zarr.group(...); root.create_dataset(<name>, ...)`,
        this will open the group and return the named dataset. If no dataset_name is provided and the group contains
        exactly one array, that array will be returned. As a fallback, if the path is a bare array store, the array
        itself is returned.
        """
        # Try opening as a group first (DirectoryStore case)
        try:
            grp = zarr.open_group(str(store_path), mode="r")
            if dataset_name is not None:
                return grp[dataset_name]
            # If no name provided, attempt to infer when there's exactly one array at root
            array_keys = list(grp.array_keys())
            if len(array_keys) == 1:
                return grp[array_keys[0]]
            raise KeyError(
                f"Dataset name is required for group at {store_path} (found arrays: {array_keys})"
            )
        except Exception:
            # Not a group or cannot find the key; try opening as a bare array
            arr = zarr.open(str(store_path), mode="r")
            # zarr.open returns either Array or Group; ensure it's an Array-like with shape
            if hasattr(arr, "shape"):
                return arr
            raise

    def list_subjects(self) -> List[str]:
        """List all subject IDs in the dataset."""
        subjects = [p.name.split('_')[1] for p in self.dataset_root.glob("Subject_*") if p.is_dir()]
        return sorted(subjects)
    
    def list_sessions(self, subject_id: str) -> List[str]:
        """List all session dates for a given subject."""
        subject_dir = self.dataset_root / f"Subject_{subject_id}"
        if not subject_dir.exists():
            raise ValueError(f"Subject {subject_id} does not exist in the dataset.")
        sessions = [p.name.split('_')[1] for p in subject_dir.glob("Session_*") if p.is_dir()]
        return sorted(sessions)
    
    def get_session_paths(self, s: SessionRef) -> Dict[str, Path]:
        """Get paths to key files in a session."""
        d = s.session_dir()
        if not d.exists():
            raise ValueError(f"Session directory {d} does not exist.")
        return {
            "traces": d / Path("traces.zarr"),
            "behaviors": d / Path("behaviors.zarr"),
            "events": d / Path("events.zarr"),
            "events_schema": d / Path("events_schema.json"),
            "behaviors_schema": d / Path("behaviors_schema.json"),
        }
    
    def get_session(self, subject_id: str, session_date: str) -> SessionRef:
        """Create a new session directory for a subject."""
        return SessionRef(self.dataset_root, subject_id, session_date)

    def get_shapes(self, s: SessionRef) -> Dict[str, int]:
        p = self.get_session_paths(s)
        # These stores were saved as groups with datasets named 'traces' and 'behaviors'
        traces = self._open_dataset(p["traces"], dataset_name="traces")
        behaviors = self._open_dataset(p["behaviors"], dataset_name="behaviors")
        T, N_cell = traces.shape
        Tb, N_behv = behaviors.shape
        if Tb != T:
            raise ValueError(f"Time mismatch: traces T={T}, behaviors T={Tb}")
        try:
            events = self._open_dataset(p["events"], dataset_name="events")
            Te, N_event = events.shape
            if Te != T: raise ValueError(f"Time mismatch: events T={Te}, traces T={T}")
        except Exception:
            N_event = 0
        return {"T": T, "N_cell": N_cell, "N_behv": N_behv, "N_event": N_event}

    def list_event_columns(self, s: SessionRef) -> List[str]:
        with open(self.get_session_paths(s)["events_schema"], "r") as f:
            schema = json.load(f)
        return [c["name"] for c in schema["columns"]]

    def list_behavior_columns(self, s: SessionRef) -> List[str]:
        with open(self.get_session_paths(s)["behaviors_schema"], "r") as f:
            schema = json.load(f)
        return [c["name"] for c in schema["columns"]]
    
    def list_brain_regions(self, s: SessionRef) -> List[str]:
        cells_info_path = s.session_dir() / "cells_info.parquet"
        if not cells_info_path.exists():
            raise ValueError(f"cells_info.parquet not found in session directory {s.session_dir()}")
        cells_info = pd.read_parquet(cells_info_path)
        if "brain_region" not in cells_info.columns:
            raise ValueError(f"'brain_region' column not found in cells_info.parquet")
        return sorted(cells_info["brain_region"].unique().tolist())
    
    # ------------------------- get window manifest -------------------------
    def compute_windows_event_window(self, s: SessionRef, align_event: str,
                                     pre: int, post: int, drop_oob: bool = False, min_trial_len=None, max_trial_len=None) -> List[Window]:
        paths = self.get_session_paths(s)
        # events are stored as a dataset named 'events' under a group
        events = self._open_dataset(paths["events"], dataset_name="events")
        cols = self.list_event_columns(s)
        align_col = cols.index(align_event)
        idxs = np.where(events[:, align_col]==1)[0]
        T = events.shape[0]

        out: List[Window] = []
        for t0 in idxs:
            a, b = int(t0 - pre), int(t0 + post)
            if 0 <= a < b <= T:
                if min_trial_len is not None and (b - a) < min_trial_len:
                    continue
                if max_trial_len is not None and (b - a) > max_trial_len:
                    continue
                out.append(Window(a, b))
            elif not drop_oob:
                raise ValueError(f"OOB window around {t0}: [{a},{b})")
        return out
    
    def compute_windows_event_range(self, s: SessionRef, start_event: str, end_event: str, drop_oob: bool = False, min_trial_len=None, max_trial_len=None) -> List[Window]:
        paths = self.get_session_paths(s)
        events = self._open_dataset(paths["events"], dataset_name="events")
        cols = self.list_event_columns(s)
        starts_col, ends_col = cols.index(start_event), cols.index(end_event)
        starts = np.where(events[:, starts_col]==1)[0]
        ends = np.where(events[:, ends_col]==1)[0]
        T = events.shape[0]
        out: List[Window] = []
        assert len(starts) == len(ends), "Mismatched start/end event counts"
        assert np.all(starts <= ends), "Start events must not occur after end events"

        for s, e in zip(starts, ends):
            a, b = int(s), int(e)
            if 0 <= a < b <= T:
                if min_trial_len is not None and (b - a) < min_trial_len:
                    continue
                if max_trial_len is not None and (b - a) > max_trial_len:
                    continue
                out.append(Window(a, b))
            elif not drop_oob:
                raise ValueError(f"OOB window around {s}-{e}: [{a},{b})")
        
        return out

    # ---------- manifests ----------
    def write_windows_manifest(self, s: SessionRef, windows: List[Window],
                               windows_name: Optional[str] = None) -> Path:
        sd = s.session_dir()
        out = Path(sd / "by_trial")
        out.mkdir(parents=True, exist_ok=True)
        man = {
            "session": {
                "root": str(self.dataset_root), "subject_id": s.subject_id, "session_date": s.session_date
            },
            "arrays": {
                "traces": str(sd / "traces.zarr"),
                "behaviors": str(sd / "behaviors.zarr"),
            },
            "windows": [{"start": w.start, "end": w.end, "length": w.length} for w in windows],
        }
        path = out / f"windows_{windows_name}.json" if windows_name else out / "windows.json"
        with open(path, "w") as f: json.dump(man, f, indent=2)
        return path

    def read_windows_manifest(self, path: Path) -> WindowsManifest:
        with open(path, "r") as f: man = json.load(f)
        s = SessionRef(Path(man["session"]["root"]),
                       man["session"]["subject_id"],
                       man["session"]["session_date"])
        windows = [Window(m["start"], m["end"]) for m in man["windows"]]
        return WindowsManifest(s, man["arrays"], windows)
    # ---------- load entire session data ----------
    def load_neural_data(self, s: SessionRef, region_keys: Optional[List[str]]=None) -> np.ndarray:
        p = self.get_session_paths(s)
        arr = self._open_dataset(p["traces"], dataset_name="traces")
        if region_keys is not None:
            cells_info = pd.read_parquet(s.session_dir() / "cells_info.parquet")
            # check if all region_keys exist
            for rk in region_keys:
                if rk not in cells_info["brain_region"].values:
                    raise ValueError(f"Region key {rk} not found in cells_info brain_region column.")
            region_mask = cells_info["brain_region"].isin(region_keys).values
            arr = arr[:, region_mask]
        return arr[:,:]  # (T, N_cell)
    
    def load_behavior_data(self, s: SessionRef, behavior_keys: Optional[List[str]]=None) -> np.ndarray:
        p = self.get_session_paths(s)
        arr = self._open_dataset(p["behaviors"], dataset_name="behaviors")
        if behavior_keys is None:
            return arr[:,:]  # (T, N_behv)
        # Find indices of requested behavior keys from the behaviors_schema.json
        behv_lst = self.list_behavior_columns(s)
        behv_idxs = []
        for k in behavior_keys:
            if k not in behv_lst:
                raise ValueError(f"Behavior key {k} not found in behaviors columns: {behv_lst}")
            behv_idxs.append(behv_lst.index(k))
        return arr[:, behv_idxs]  # (T, len(behavior_keys))
    
    def load_cells_info(self, s: SessionRef) -> pd.DataFrame:
        cells_info_path = s.session_dir() / "cells_info.parquet"
        if not cells_info_path.exists():
            raise ValueError(f"cells_info.parquet not found in session directory {s.session_dir()}")
        cells_info = pd.read_parquet(cells_info_path)
        return cells_info
    
    def load_events(self, s: SessionRef) -> np.ndarray:
        p = self.get_session_paths(s)
        events = self._open_dataset(p["events"], dataset_name="events")
        return events[:,:]  # (T, N_event)
    
    # ---------- get by trial data ----------
    def load_neural_trials(self, manifest: WindowsManifest, region_keys: Optional[List[str]]=None) -> List[np.ndarray]:
        arr = self._open_dataset(manifest.arrays["traces"], dataset_name="traces")
        # create region mask if region_keys provided
        if region_keys is not None:
            cells_info = pd.read_parquet(manifest.session.session_dir() / "cells_info.parquet")
            # check if all region_keys exist
            for rk in region_keys:
                if rk not in cells_info["brain_region"].values:
                    raise ValueError(f"Region key {rk} not found in cells_info brain_region column.")
            region_mask = cells_info["brain_region"].isin(region_keys).values
            arr = arr[:, region_mask]
        traces_bytrial = []
        for w in manifest.windows:
            traces_bytrial.append(arr[w.start:w.end, :])
        return traces_bytrial

    def load_behavior_trials(self, manifest: WindowsManifest, behavior_keys: Optional[List[str]]=None) -> List[np.ndarray]:
        arr = self._open_dataset(manifest.arrays["behaviors"], dataset_name="behaviors")
        behvs_bytrial = []
        # Find indices of requested behavior keys from the behaviors_schema.json
        behv_lst = self.list_behavior_columns(manifest.session)
        if behavior_keys is None:
            behv_idxs = np.arange(arr.shape[1])
        else:
            behv_idxs = []
            for k in behavior_keys:
                if k not in behv_lst:
                    raise ValueError(f"Behavior key {k} not found in behaviors columns: {behv_lst}")
                behv_idxs.append(behv_lst.index(k))
            behv_idxs = np.array(behv_idxs)
        for w in manifest.windows:
            behvs_bytrial.append(arr[w.start:w.end, behv_idxs])
        return behvs_bytrial
