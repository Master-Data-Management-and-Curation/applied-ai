"""
Data loader for Sensorium dashboard.
Handles loading data from multiple mice, metadata, and coordinating data access.
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .loader import SensoriumSession


class MouseDataManager:
    """
    Manages data loading for multiple mice.
    Handles metadata JSON files, cell_motor_coordinates, and video/response/behavior/pupil data.
    """
    
    def __init__(self, project_data_dir: str, results_dir: str):
        """
        Initialize the MouseDataManager.
        
        Args:
            project_data_dir: Path to project_data directory containing mouse folders
            results_dir: Path to results directory containing metadata JSON files
        """
        self.project_data_dir = os.path.abspath(project_data_dir)
        self.results_dir = os.path.abspath(results_dir)
        
        # Cache for loaded data
        self._mice_cache: Dict[str, Dict[str, Any]] = {}
        self._sessions_cache: Dict[str, SensoriumSession] = {}
        self._metadata_cache: Dict[str, Dict] = {}
        self._cell_coords_cache: Dict[str, np.ndarray] = {}
        
        # Discover available mice
        self._discover_mice()
    
    def _discover_mice(self) -> None:
        """Discover all available mice in project_data directory."""
        if not os.path.isdir(self.project_data_dir):
            raise ValueError(f"Project data directory not found: {self.project_data_dir}")
        
        self.available_mice = []
        for item in os.listdir(self.project_data_dir):
            item_path = os.path.join(self.project_data_dir, item)
            # Skip zip files and __MACOSX directory
            if os.path.isdir(item_path) and not item.startswith("__MACOSX") and not item.endswith(".zip"):
                self.available_mice.append(item)
        
        self.available_mice.sort()
    
    def get_available_mice(self) -> List[str]:
        """Get list of available mouse IDs."""
        return self.available_mice.copy()
    
    def _get_mouse_path(self, mouse_id: str) -> str:
        """Get full path to mouse directory."""
        return os.path.join(self.project_data_dir, mouse_id)
    
    def _get_metadata_path(self, mouse_id: str) -> str:
        """Get path to metadata JSON file for a mouse."""
        return os.path.join(self.results_dir, f"combined_metadata_{mouse_id}.json")
    
    def _get_cell_coords_path(self, mouse_id: str) -> str:
        """Get path to cell_motor_coordinates file for a mouse."""
        mouse_path = self._get_mouse_path(mouse_id)
        return os.path.join(mouse_path, "meta", "neurons", "cell_motor_coordinates.npy")
    
    def _load_session(self, mouse_id: str) -> SensoriumSession:
        """Load or get cached SensoriumSession for a mouse."""
        if mouse_id not in self._sessions_cache:
            mouse_path = self._get_mouse_path(mouse_id)
            if not os.path.isdir(mouse_path):
                raise ValueError(f"Mouse directory not found: {mouse_path}")
            self._sessions_cache[mouse_id] = SensoriumSession(mouse_path)
        return self._sessions_cache[mouse_id]
    
    def _load_metadata(self, mouse_id: str) -> Dict:
        """Load metadata JSON file for a mouse."""
        if mouse_id not in self._metadata_cache:
            metadata_path = self._get_metadata_path(mouse_id)
            if not os.path.exists(metadata_path):
                raise ValueError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                self._metadata_cache[mouse_id] = json.load(f)
        
        return self._metadata_cache[mouse_id]
    
    def get_cell_motor_coordinates(self, mouse_id: str) -> Optional[np.ndarray]:
        """
        Load cell_motor_coordinates for a mouse.
        
        Args:
            mouse_id: Mouse identifier
            
        Returns:
            numpy array of shape (N, 3) or None if not found
        """
        if mouse_id not in self._cell_coords_cache:
            coords_path = self._get_cell_coords_path(mouse_id)
            if os.path.exists(coords_path):
                self._cell_coords_cache[mouse_id] = np.load(coords_path)
            else:
                self._cell_coords_cache[mouse_id] = None
        
        return self._cell_coords_cache[mouse_id]
    
    def get_representative_videos(self, mouse_id: str) -> List[str]:
        """
        Get list of representative video IDs for a mouse.
        
        Args:
            mouse_id: Mouse identifier
            
        Returns:
            List of representative video IDs (as strings)
        """
        metadata = self._load_metadata(mouse_id)
        return sorted(metadata.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    
    def get_video_metadata(self, mouse_id: str, video_id: str) -> Dict:
        """
        Get metadata for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (representative video ID)
            
        Returns:
            Dictionary with video metadata including equivalent_videos, video_valid_frames, etc.
        """
        metadata = self._load_metadata(mouse_id)
        return metadata.get(str(video_id), {})
    
    def get_equivalent_videos(self, mouse_id: str, representative_video_id: str) -> List[str]:
        """
        Get list of equivalent video IDs for a representative video.
        
        Args:
            mouse_id: Mouse identifier
            representative_video_id: Representative video ID
            
        Returns:
            List of equivalent video IDs (as strings)
        """
        video_metadata = self.get_video_metadata(mouse_id, representative_video_id)
        return video_metadata.get("equivalent_videos", [])
    
    def get_all_video_ids_for_plotting(self, mouse_id: str, representative_video_id: str) -> List[str]:
        """
        Get list of all video IDs to plot (representative + equivalent).
        
        Args:
            mouse_id: Mouse identifier
            representative_video_id: Representative video ID
            
        Returns:
            List of video IDs: [representative_video_id, ...equivalent_video_ids]
        """
        equivalent = self.get_equivalent_videos(mouse_id, representative_video_id)
        return [str(representative_video_id)] + [str(vid) for vid in equivalent]
    
    def get_video_data(self, mouse_id: str, video_id: str) -> Optional[np.ndarray]:
        """
        Get video data (numpy array) for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (as string or int)
            
        Returns:
            numpy array of video frames or None if not found
        """
        session = self._load_session(mouse_id)
        video_id_int = int(video_id) if isinstance(video_id, str) else video_id
        return session.get_video_frames(video_id_int)
    
    def get_responses_data(self, mouse_id: str, video_id: str) -> Optional[np.ndarray]:
        """
        Get responses data for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (as string or int)
            
        Returns:
            numpy array of shape (n_neurons, n_frames) or None if not found
        """
        session = self._load_session(mouse_id)
        video_id_int = int(video_id) if isinstance(video_id, str) else video_id
        return session.get_responses(video_id_int)
    
    def get_behavior_data(self, mouse_id: str, video_id: str) -> Optional[np.ndarray]:
        """
        Get behavior data for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (as string or int)
            
        Returns:
            numpy array of shape (2, n_frames) where [0] is running speed, [1] is pupil dilation
        """
        session = self._load_session(mouse_id)
        video_id_int = int(video_id) if isinstance(video_id, str) else video_id
        return session.get_behavior(video_id_int)
    
    def get_pupil_center_data(self, mouse_id: str, video_id: str) -> Optional[np.ndarray]:
        """
        Get pupil center data for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (as string or int)
            
        Returns:
            numpy array of shape (2, n_frames) where [0] is horizontal, [1] is vertical position
        """
        session = self._load_session(mouse_id)
        video_id_int = int(video_id) if isinstance(video_id, str) else video_id
        return session.get_pupil_center(video_id_int)
    
    def get_num_neurons(self, mouse_id: str, video_id: str) -> int:
        """
        Get number of neurons for a specific video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Video ID (as string or int)
            
        Returns:
            Number of neurons
        """
        responses = self.get_responses_data(mouse_id, video_id)
        if responses is None:
            return 0
        responses = np.asarray(responses)
        if responses.ndim == 2:
            # Assume (n_neurons, n_frames) or (n_frames, n_neurons)
            return responses.shape[0] if responses.shape[0] > responses.shape[1] else responses.shape[1]
        return 0
    
    def get_video_info(self, mouse_id: str, video_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a video.
        
        Args:
            mouse_id: Mouse identifier
            video_id: Representative video ID
            
        Returns:
            Dictionary with video information including:
            - video_id: Video ID
            - video_valid_frames: Number of valid frames
            - number_equivalent_videos: Number of equivalent videos
            - equivalent_videos: List of equivalent video IDs
            - same_valid_responses: Whether responses are consistent across equivalent videos
            - incorrect_valid_responses: Number of incorrect responses on valid frames
        """
        metadata = self.get_video_metadata(mouse_id, video_id)
        return {
            "video_id": str(video_id),
            "video_valid_frames": metadata.get("video_valid_frames", 0),
            "number_equivalent_videos": metadata.get("number_equivalent_videos", 0),
            "equivalent_videos": metadata.get("equivalent_videos", []),
            "same_valid_responses": metadata.get("same_valid_responses", False),
            "incorrect_valid_responses": metadata.get("incorrect_valid_responses", 0),
        }





