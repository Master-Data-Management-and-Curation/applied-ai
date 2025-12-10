"""
Django views for Sensorium webapp.
"""
import os
import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import sys

# Import data loader from same directory
from .data_loader import MouseDataManager

# Initialize data manager
# Use Django's BASE_DIR from settings for consistency
from django.conf import settings
BASE_DIR = settings.BASE_DIR

# Convert Path object to string if needed
if hasattr(BASE_DIR, '__str__'):
    BASE_DIR = str(BASE_DIR)

# Paths are relative to BASE_DIR, making the project portable
# These paths will work regardless of where the project directory is located
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "project_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Verify paths exist (only warn, don't fail at import time)
if not os.path.isdir(PROJECT_DATA_DIR):
    import warnings
    warnings.warn(f"Project data directory not found: {PROJECT_DATA_DIR}. Please ensure 'project_data' folder exists in the project root.")
if not os.path.isdir(RESULTS_DIR):
    import warnings
    warnings.warn(f"Results directory not found: {RESULTS_DIR}. Please ensure 'results' folder exists in the project root.")

data_manager = MouseDataManager(PROJECT_DATA_DIR, RESULTS_DIR)


def index(request):
    """Main dashboard view."""
    available_mice = data_manager.get_available_mice()
    
    # Set default values
    default_mouse = available_mice[0] if available_mice else None
    default_video = None
    default_neuron = 0
    
    if default_mouse:
        try:
            representative_videos = data_manager.get_representative_videos(default_mouse)
            if representative_videos:
                # Try to find a video with equivalent videos for better demo
                for vid in representative_videos:
                    eq_videos = data_manager.get_equivalent_videos(default_mouse, vid)
                    if len(eq_videos) > 0:
                        default_video = vid
                        break
                # If none found, use first video
                if default_video is None:
                    default_video = representative_videos[0]
        except:
            pass
    
    context = {
        'available_mice': available_mice,
        'default_mouse': default_mouse,
        'default_video': default_video,
        'default_neuron': default_neuron,
    }
    return render(request, 'webapp/index.html', context)


@csrf_exempt
def api_get_videos(request, mouse_id):
    """API endpoint to get representative videos for a mouse."""
    try:
        videos = data_manager.get_representative_videos(mouse_id)
        return JsonResponse({'videos': videos})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_get_video_info(request, mouse_id, video_id):
    """API endpoint to get video information."""
    try:
        video_info = data_manager.get_video_info(mouse_id, video_id)
        equivalent_videos = data_manager.get_equivalent_videos(mouse_id, video_id)
        video_info['equivalent_videos'] = equivalent_videos
        return JsonResponse(video_info)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_get_neurons(request, mouse_id, video_id):
    """API endpoint to get number of neurons for a video."""
    try:
        num_neurons = data_manager.get_num_neurons(mouse_id, video_id)
        return JsonResponse({'num_neurons': num_neurons})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_get_plot_data(request, mouse_id, video_id, data_type):
    """
    API endpoint to get plot data (responses, behavior, or pupil_center).
    data_type: 'responses', 'behavior', or 'pupil_center'
    """
    try:
        # Get all video IDs to plot (representative + equivalent)
        all_video_ids = data_manager.get_all_video_ids_for_plotting(mouse_id, video_id)
        
        plot_data = {}
        for vid_id in all_video_ids:
            if data_type == 'responses':
                data = data_manager.get_responses_data(mouse_id, vid_id)
            elif data_type == 'behavior':
                data = data_manager.get_behavior_data(mouse_id, vid_id)
            elif data_type == 'pupil_center':
                data = data_manager.get_pupil_center_data(mouse_id, vid_id)
            else:
                return JsonResponse({'error': 'Invalid data_type'}, status=400)
            
            if data is not None:
                # Convert numpy array to list for JSON serialization
                # Replace NaN values with null for JSON compatibility
                if isinstance(data, np.ndarray):
                    data_list = data.tolist()
                    # Recursively replace NaN with None (which becomes null in JSON)
                    def replace_nan(obj):
                        if isinstance(obj, list):
                            return [replace_nan(item) for item in obj]
                        elif isinstance(obj, float) and np.isnan(obj):
                            return None
                        else:
                            return obj
                    plot_data[str(vid_id)] = replace_nan(data_list)
                else:
                    plot_data[str(vid_id)] = data
        
        return JsonResponse({
            'video_ids': [str(vid) for vid in all_video_ids],
            'plot_data': plot_data
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_get_cell_coordinates(request, mouse_id):
    """API endpoint to get cell motor coordinates."""
    try:
        coords = data_manager.get_cell_motor_coordinates(mouse_id)
        if coords is not None:
            # Replace NaN values with null for JSON compatibility
            coords_list = coords.tolist()
            def replace_nan(obj):
                if isinstance(obj, list):
                    return [replace_nan(item) for item in obj]
                elif isinstance(obj, float) and np.isnan(obj):
                    return None
                else:
                    return obj
            coords_clean = replace_nan(coords_list)
            return JsonResponse({'coordinates': coords_clean})
        else:
            return JsonResponse({'error': 'Cell coordinates not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_get_video_base64(request, mouse_id, video_id):
    """API endpoint to get video as base64 encoded data."""
    try:
        video_array = data_manager.get_video_data(mouse_id, video_id)
        if video_array is None:
            return JsonResponse({'error': 'Video not found'}, status=404)
        
        # Convert to base64 using utility
        from .utils.video_converter import numpy_to_video_base64, FFMPEG_AVAILABLE, IMAGEIO_AVAILABLE
        
        # Check if required dependencies are available
        if not IMAGEIO_AVAILABLE or not FFMPEG_AVAILABLE:
            error_msg = 'Video conversion requires imageio and imageio-ffmpeg. '
            if not IMAGEIO_AVAILABLE:
                error_msg += 'Please install: pip install imageio imageio-ffmpeg'
            elif not FFMPEG_AVAILABLE:
                error_msg += 'Please install: pip install imageio-ffmpeg'
            return JsonResponse({'error': error_msg}, status=500)
        
        video_base64 = numpy_to_video_base64(video_array, fps=30)
        if video_base64:
            # Extract base64 part (remove data URI prefix)
            if video_base64.startswith('data:video/mp4;base64,'):
                video_base64 = video_base64.split(',', 1)[1]
            return JsonResponse({'video_data': video_base64})
        else:
            return JsonResponse({'error': 'Failed to convert video. Check server logs for details.'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
