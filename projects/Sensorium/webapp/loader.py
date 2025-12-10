# loader.py
import os
import numpy as np
from glob import glob

class SensoriumSession:
    """
    Lightweight loader for one session directory.
    It expects `data/{videos,responses,behavior,pupil_center}` under root_dir.
    It maps files by integer-stem filenames (0.npy, 1.npy, ...).
    """

    def __init__(self, root_dir):
        self.root = os.path.abspath(root_dir)
        self.data_dir = os.path.join(self.root, "data")
        # folders (if missing, set to None)
        self.folders = {
            "videos": os.path.join(self.data_dir, "videos"),
            "responses": os.path.join(self.data_dir, "responses"),
            "behavior": os.path.join(self.data_dir, "behavior"),
            "pupil_center": os.path.join(self.data_dir, "pupil_center"),
        }
        self.index = {}
        self._build_index()

    def _list_files(self, folder):
        if not folder or not os.path.isdir(folder):
            return []
        # accept .npy, .npz, .mp4, .avi, .mov
        exts = ("*.npy", "*.npz", "*.mp4", "*.avi", "*.mov")
        files = []
        for e in exts:
            files += glob(os.path.join(folder, e))
        return sorted(files, key=lambda p: os.path.basename(p))

    def _stem_int(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name)
        except Exception:
            return None

    def _build_index(self):
        # map trial_id -> file path for each mod
        # self.index = { trial_id: {"behavior":path, "videos":path, ...} }
        idx = {}
        for mod, folder in self.folders.items():
            files = self._list_files(folder)
            for f in files:
                s = self._stem_int(f)
                if s is None:
                    # skip non-integer filenames (could extend to metadata mapping)
                    continue
                if s not in idx:
                    idx[s] = {}
                idx[s][mod] = f
        # convert to dict with contiguous sorted keys
        self.index = {k: idx[k] for k in sorted(idx.keys())}
        self.trial_ids = sorted(self.index.keys())
        # quick counts
        self.n_trials = len(self.trial_ids)

    def list_trials(self):
        return self.trial_ids

    def _load_any(self, path):
        # load depending on extension
        if path is None:
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            return np.load(path, allow_pickle=True)
        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            # return dict-like
            return dict(data)
        # for video files we won't attempt to load entire mp4 (too heavy)
        # instead return the path and let the app use OpenCV to read frames on demand
        if ext in (".mp4", ".avi", ".mov"):
            return path
        # fallback: try numpy
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Unsupported file type {path}: {e}")

    def get_behavior(self, trial_id):
        p = self.index.get(trial_id, {}).get("behavior")
        if p is None: return None
        return self._load_any(p)

    def get_responses(self, trial_id):
        p = self.index.get(trial_id, {}).get("responses")
        if p is None: return None
        return self._load_any(p)

    def get_pupil_center(self, trial_id):
        p = self.index.get(trial_id, {}).get("pupil_center")
        if p is None: return None
        return self._load_any(p)

    def get_video_path(self, trial_id):
        p = self.index.get(trial_id, {}).get("videos")
        if p is None: return None
        return p

    def get_video_frames(self, trial_id, max_frames=None):
        """
        Loads video frames into numpy array if video file is .npy or .npz,
        otherwise returns video file path (for streaming via cv2).
        """
        p = self.get_video_path(trial_id)
        if p is None:
            return None
        ext = os.path.splitext(p)[1].lower()
        if ext == ".npy":
            v = np.load(p, allow_pickle=True)
            if max_frames is not None:
                return v[:max_frames]
            return v
        if ext == ".npz":
            data = np.load(p, allow_pickle=True)
            # if array stored under some key, try first key
            keys = list(data.keys())
            if keys:
                v = data[keys[0]]
                return v if max_frames is None else v[:max_frames]
            return None
        # for mp4/avi return path and let frontend read frames on demand
        return p

    def get_aligned(self, trial_id):
        """
        Returns a dictionary:
         { "behavior": arr, "responses": arr, "pupil_center": arr, "video": arr or path }
        """
        return {
            "behavior": self.get_behavior(trial_id),
            "responses": self.get_responses(trial_id),
            "pupil_center": self.get_pupil_center(trial_id),
            "video": self.get_video_frames(trial_id),
        }
