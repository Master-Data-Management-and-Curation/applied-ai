import os
import numpy as np
import json
import time
from typing import Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class Exploration:
    def __init__(self, mouse_id_path):
        self.mouse_id_path = mouse_id_path
        self._videos_path = os.path.join(self.mouse_id_path, "data/videos")
        self._responses_path = os.path.join(self.mouse_id_path, "data/responses")

    def _count_valid_frames_per_video(
        self, videos_filenames: Optional[Dict[str, int]] = None
    ):
        valid_frame_counts = {}
        if not videos_filenames:
            videos_list = os.listdir(self._videos_path)
        else:
            videos_list = videos_filenames
        for video_file in videos_list:
            video = np.load(os.path.join(self._videos_path, video_file))
            # video has shape (36, 64, 324), with 324 the number of frames
            # for each video we count the number of valid frames (non-NaN)
            valid_frames = np.sum(~np.isnan(video[0, 0, :]))
            valid_frame_counts[video_file.replace(".npy", "")] = int(valid_frames)
        return valid_frame_counts

    def _validate_neuron_responses_per_video(
        self, responses_filenames: Optional[Dict[str, int]] = None
    ):
        """Count the number of frames with valid neuron responses for each video."""
        valid_response_counts = {}
        if not responses_filenames:
            responses_list = os.listdir(self._responses_path)
        else:
            responses_list = responses_filenames
        for response_file in responses_list:
            response = np.load(os.path.join(self._responses_path, response_file))
            # response has shape (N_neurons, N_frames), we count non-NaN responses
            valid_frames = np.sum(~np.isnan(response), axis=0)  # Count per neuron
            number_frames_correct_responses = np.sum(
                ~np.isnan(valid_frames)
            )  # Count overall valid frames
            valid_response_counts[response_file.replace(".npy", "")] = int(
                number_frames_correct_responses
            )
        return valid_response_counts

    def _find_unique_videos_and_representative_relations(
        self,
        method="cosine",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        cosine_threshold=0.98,
        save_path=None,
    ):
        """
        Find unique videos and their representative relations.

        The cosine similarity method is the most effective but also the most computationally expensive.
        The current version works but is less than optimal in terms of speed and memory usage.

        Args:
            method (str): Comparison method - "exact", "cosine", or "allclose
            rtol (float): Relative tolerance for "allclose" method.
            atol (float): Absolute tolerance for "allclose" method.
            cosine_threshold (float): Threshold for cosine similarity method.
            save_path (str): Path to save the results as a JSON file.
        """
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if method not in ["exact", "cosine", "allclose"]:
            raise ValueError("Method must be one of 'exact', 'cosine', or 'allclose'.")

        cmp_func = None
        if rtol is not None and atol is not None and method == "allclose":

            def video_allclose(x, y):
                return np.allclose(x, y, rtol=rtol, atol=atol)

            cmp_func = video_allclose
        elif method == "cosine":

            def video_cosine_similarity(x, y):
                # Reshape to 2D arrays for cosine similarity computation
                x_reshaped = x.ravel().reshape(1, -1)
                y_reshaped = y.ravel().reshape(1, -1)
                similarity = cosine_similarity(x_reshaped, y_reshaped)
                # Check if all frame similarities exceed the threshold
                return np.all(similarity > cosine_threshold)

            cmp_func = video_cosine_similarity
        else:
            cmp_func = np.array_equal

        videos_list = os.listdir(self._videos_path)
        videos_data = {
            video_file.replace(".npy", ""): np.load(
                os.path.join(self._videos_path, video_file)
            )
            for video_file in videos_list
        }
        # remove non NaN frames for comparison
        for video in videos_data.keys():
            video_data = videos_data[video]
            # if an image frame has NaN values, all frames are NaN and thus here
            # we just filter based on the first pixel
            valid_frames = ~np.isnan(video_data[0, 0, :])
            videos_data[video] = video_data[:, :, valid_frames]

        videos_relations = {}
        processed_as_duplicate = set()  # Track videos already found as duplicates

        print("Starting comparison of videos to find unique ones...")

        for video in videos_data.keys():
            # Skip if this video was already found as a duplicate of another video
            if video in processed_as_duplicate:
                continue

            # Initialize relations list for this representative video
            videos_relations[video] = []

            # Compare with all other unprocessed videos
            for other_video in videos_data.keys():
                if video == other_video or other_video in processed_as_duplicate:
                    continue
                # compare videos data element-wise
                video1 = videos_data[video]
                video2 = videos_data[other_video]
                if (
                    video1.shape[2] == video2.shape[2]  # Ensure same number of frames
                    and cmp_func(video1, video2)  # Compare if the videos are similar
                ):
                    videos_relations[video].append(other_video)  # Record as duplicate
                    processed_as_duplicate.add(other_video)  # Mark as duplicate

        # Sort the videos_relations by keys numerically and values lists numerically
        videos_relations = {
            k: sorted(v, key=lambda x: int(x))
            for k, v in sorted(videos_relations.items(), key=lambda item: int(item[0]))
        }

        number_of_videos = len(videos_list)
        number_of_unique_videos = len(videos_relations)
        number_of_repeated_videos = number_of_videos - number_of_unique_videos
        print(f"There are {number_of_videos} videos in total.")
        print(f"There are {number_of_unique_videos} unique videos.")
        print(f"There are {number_of_repeated_videos} repeated videos.")

        if save_path:
            print(f"Saving unique videos relations to {save_path}...")
            with open(save_path, "w") as f:
                json.dump(videos_relations, f, indent=4)

        return videos_relations

    def analyze_videos_and_responses(
        self,
        method="cosine",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        cosine_threshold=0.98,
        save_path=None,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Check that for each video, the number of valid frames matches the number
        of valid neuron responses.
        Also, group videos into unique ones and their representative relations.

        Args:
            save_path (str): Path to save the analysis report as a JSON file.
        """
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Find unique videos and their representative relations
        unique_videos_relations = (
            self._find_unique_videos_and_representative_relations(
                method=method,
                rtol=rtol,
                atol=atol,
                cosine_threshold=cosine_threshold,
            )
        )
        # Given the unique videos, count valid frames and validate neuron responses
        unique_videos_filenames = [
            f"{video}.npy" for video in unique_videos_relations.keys()
        ]
        videos_valid_frames = self._count_valid_frames_per_video(
            videos_filenames=unique_videos_filenames
        )
        responses_validation = self._validate_neuron_responses_per_video(
            responses_filenames=unique_videos_filenames
        )

        analysis_report = {}

        for representative_video in unique_videos_relations.keys():
            number_valid_video_frames = videos_valid_frames[representative_video]
            number_valid_video_responses = responses_validation[representative_video]
            correct_neuron_responses = (
                number_valid_video_frames == number_valid_video_responses
            )
            incorrect_neuron_responses = abs(
                number_valid_video_frames - number_valid_video_responses
            )
            equivalent_videos = unique_videos_relations[representative_video]
            analysis_report[representative_video] = {
                "number_equivalent_videos": len(equivalent_videos),
                "equivalent_videos": equivalent_videos,
                "video_valid_frames": number_valid_video_frames,
                "same_valid_responses": correct_neuron_responses,
                "incorrect_valid_responses": incorrect_neuron_responses,
                "label": labels.get(representative_video, "Unknown") if labels else "Unknown",
            }

        if save_path:
            with open(save_path, "w") as f:
                json.dump(analysis_report, f, indent=4)

        return analysis_report


def process_single_mouse(
    mouse_id_path, 
    method="cosine",
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    cosine_threshold=0.98,
    ):
    """Process a single mouse dataset."""
    exploration = Exploration(mouse_id_path)
    start_time = time.time()
    mouse_folder = os.path.basename(mouse_id_path)
    print(f"[{mouse_folder}] Analyzing videos and responses...")
    
    # Load labels from classification table
    labels = {}
    try:
        df = pd.read_csv("results/classification_table_naming.csv")
        mouse_data = df[df['recording'] == mouse_folder]
        for _, row in mouse_data.iterrows():
            video_id = row['file'].replace('.npy', '')
            labels[video_id] = row['label']
    except FileNotFoundError:
        print("Warning: classification_table_naming.csv not found, labels will be 'Unknown'")
    
    exploration.analyze_videos_and_responses(
        method=method,
        rtol=rtol,
        atol=atol,
        cosine_threshold=cosine_threshold,
        save_path=f"results/combined_metadata_{mouse_folder}.json",
        labels=labels
    )
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"[{mouse_folder}] Time taken: {total_time_minutes:.2f} minutes")
    return mouse_folder, total_time_minutes
