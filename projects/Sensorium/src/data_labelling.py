import os
import numpy as np
import math
from scipy.spatial.distance import cdist
from dadapy.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
from sklearn.cluster import KMeans

MOUSE_INFORMATION_DIR = "/orfeo/scratch/mdmc/vnkanang/Sensorium/project_data/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce"
VIDEOS_DIR = "/orfeo/scratch/mdmc/vnkanang/Sensorium/project_data/dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce/data/videos"

def load_videos(videos_dir):
    """Load videos from the specified directory."""
    videos_list = os.listdir(videos_dir)
    videos_data = {
        video_file.replace(".npy", ""): np.load(os.path.join(videos_dir, video_file))
        for video_file in videos_list
    }
    return videos_data

def preprocess_videos(videos_data, min_frames=50):
    """Preprocess videos: remove NaN frames, truncate to min_frames, normalize."""
    minimum_number_of_frames = np.inf
    for video in videos_data.keys():
        video_data = videos_data[video]
        valid_frames = ~np.isnan(video_data[0, 0, :])
        videos_data[video] = video_data[:, :, valid_frames]
        if videos_data[video].shape[2] < minimum_number_of_frames:
            minimum_number_of_frames = videos_data[video].shape[2]
    
    minimum_number_of_frames = min(minimum_number_of_frames, min_frames)
    for video in videos_data.keys():
        videos_data[video] = videos_data[video][:, :, :minimum_number_of_frames]
    
    videos_np = np.stack([videos_data[video] for video in sorted(videos_data.keys())], axis=0)
    mean = np.nanmean(videos_np)
    std = np.nanstd(videos_np)
    videos_norm = (videos_np - mean) / (std + 1e-8)
    
    # Remove duplicates
    unique_videos, unique_indices = np.unique(videos_norm, axis=0, return_index=True)
    return unique_videos, unique_indices, videos_data

def compute_distances(videos_flattened):
    """Compute cosine distances between flattened videos."""
    return cdist(videos_flattened, videos_flattened, metric='cosine')

def compute_intrinsic_dimension(distances):
    """Compute intrinsic dimension using dadapy."""
    data_dadapy = Data(distances=distances)
    id_list, _, _ = data_dadapy.return_id_scaling_gride()
    id_index = int(math.log2(16))
    data_dadapy.set_id(id_list[id_index])
    return data_dadapy

def cluster_videos(data_dadapy, videos_flattened, method='adp', n_clusters=None):
    """Cluster videos using specified method."""
    if method == 'adp':
        data_dadapy.compute_density_kNN()
        clusters = data_dadapy.compute_clustering_ADP(Z=1.645)
    elif method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters must be specified for KMeans.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(videos_flattened)
    else:
        raise ValueError("Unsupported clustering method. Choose 'adp' or 'kmeans'.")
    return clusters

def select_sample_videos(clusters, unique_indices, videos_data):
    """Select one sample video per cluster."""
    unique_labels = np.unique(clusters)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(clusters):
        label_to_indices[label].append(unique_indices[idx])
    
    sample_videos = []
    sample_titles = []
    sorted_keys = sorted(videos_data.keys())
    for label in sorted(unique_labels):
        if label_to_indices[label]:
            sample_index = label_to_indices[label][0]
            video_id = sorted_keys[sample_index]
            video = videos_data[video_id]
            sample_videos.append(video)
            sample_titles.append(f"ID: {video_id}, Label: {label}")
    return sample_videos, sample_titles

def display_multiple_videos_and_save(videos, titles, save_path, interval_ms=33):
    """Display multiple videos and save as GIF."""
    n_videos = len(videos)
    fig, axes = plt.subplots(1, n_videos, figsize=(4*n_videos, 3))
    if n_videos == 1:
        axes = [axes]
    
    imgs = []
    for i, (video, title, ax) in enumerate(zip(videos, titles, axes)):
        img = ax.imshow(video[:, :, 0], cmap="gray", animated=True)
        ax.axis("off")
        ax.set_title(title)
        imgs.append(img)
    
    num_frames = min(video.shape[2] for video in videos)
    
    def update(frame_idx):
        for img, video in zip(imgs, videos):
            img.set_data(video[:, :, frame_idx])
        return imgs
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval_ms,
        blit=True,
    )
    
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=1000/interval_ms)
    print("GIF saved successfully!")
    plt.close(fig)

# Main execution
if __name__ == "__main__":
    # Load and preprocess videos
    videos_data = load_videos(VIDEOS_DIR)
    unique_videos, unique_indices, videos_data = preprocess_videos(videos_data)
    
    # Flatten videos for distance computation
    videos_flattened = unique_videos.reshape(unique_videos.shape[0], -1)
    
    # Run ADP clustering
    print("Running ADP clustering...")
    distances = compute_distances(videos_flattened)
    data_dadapy = compute_intrinsic_dimension(distances)
    clusters_adp = cluster_videos(data_dadapy, videos_flattened, method='adp', n_clusters=None)
    unique_labels_adp = np.unique(clusters_adp)
    print(f"ADP: Number of unique clusters found: {len(unique_labels_adp)}")
    print(f"ADP: Unique cluster labels: {unique_labels_adp}")
    
    # Select sample videos for ADP
    sample_videos_adp, sample_titles_adp = select_sample_videos(clusters_adp, unique_indices, videos_data)
    
    # Save the ADP GIF
    gif_path_adp = os.path.join(os.path.dirname(__file__), "cluster_sample_videos_adp.gif")
    display_multiple_videos_and_save(sample_videos_adp, sample_titles_adp, gif_path_adp)
    
    # Run KMeans clustering with 6 clusters
    print("Running KMeans clustering with 6 clusters...")
    clusters_kmeans = cluster_videos(None, videos_flattened, method='kmeans', n_clusters=6)
    unique_labels_kmeans = np.unique(clusters_kmeans)
    print(f"KMeans: Number of unique clusters: {len(unique_labels_kmeans)}")
    print(f"KMeans: Unique cluster labels: {unique_labels_kmeans}")
    
    # Select sample videos for KMeans
    sample_videos_kmeans, sample_titles_kmeans = select_sample_videos(clusters_kmeans, unique_indices, videos_data)
    
    # Save the KMeans GIF
    gif_path_kmeans = os.path.join(os.path.dirname(__file__), "cluster_sample_videos_kmeans.gif")
    display_multiple_videos_and_save(sample_videos_kmeans, sample_titles_kmeans, gif_path_kmeans)
