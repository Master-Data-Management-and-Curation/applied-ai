"""
Utility functions for converting numpy video arrays to playable video formats.
"""
import os
import base64
import numpy as np
from typing import Optional
import tempfile
from io import BytesIO

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
    # Check if ffmpeg plugin is available
    try:
        import imageio_ffmpeg
        FFMPEG_AVAILABLE = True
    except ImportError:
        FFMPEG_AVAILABLE = False
except ImportError:
    IMAGEIO_AVAILABLE = False
    FFMPEG_AVAILABLE = False


def numpy_to_video_base64(video_array: np.ndarray, fps: int = 30) -> Optional[str]:
    """
    Convert numpy video array to base64-encoded video data URI.
    Uses imageio with ffmpeg for browser-compatible H.264 encoding if available,
    otherwise falls back to OpenCV.
    
    Args:
        video_array: numpy array of shape (n_frames, height, width, channels) or (n_frames, height, width)
                    or (height, width, frames)
        fps: frames per second for the video
        
    Returns:
        Base64-encoded video data URI string or None if conversion fails
    """
    try:
        # Handle NaN values - replace with 0
        if np.isnan(video_array).any():
            video_array = np.nan_to_num(video_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Handle video shape - convert from (height, width, frames) to (frames, height, width) if needed
        if video_array.ndim == 3:
            h, w, f = video_array.shape
            # If last dimension is largest, assume it's frames and transpose
            if f > h and f > w:
                # Shape is (height, width, frames) -> transpose to (frames, height, width)
                video_array = np.transpose(video_array, (2, 0, 1))
        
        # Normalize video array to uint8
        if video_array.dtype != np.uint8:
            # Normalize to 0-255 range
            vmin, vmax = video_array.min(), video_array.max()
            if vmax > vmin:
                video_array = ((video_array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                video_array = np.zeros_like(video_array, dtype=np.uint8)
        
        # Ensure correct shape for RGB conversion
        if video_array.ndim == 3:
            # Grayscale: (n_frames, height, width) -> add channel dimension
            video_array = video_array[..., np.newaxis]
            video_array = np.repeat(video_array, 3, axis=-1)  # Convert to RGB
        
        n_frames, height, width, channels = video_array.shape
        
        # Try imageio with ffmpeg first (better browser compatibility with H.264)
        if IMAGEIO_AVAILABLE and FFMPEG_AVAILABLE:
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                # Use imageio with ffmpeg plugin for H.264 encoding
                # This creates browser-compatible MP4 videos
                # Use the ffmpeg plugin explicitly
                writer = imageio.get_writer(tmp_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
                for frame in video_array:
                    writer.append_data(frame)
                writer.close()
                
                # Verify file was created and has content
                if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    raise RuntimeError("imageio created empty video file")
                
                # Read video file and encode to base64
                with open(tmp_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Create data URI
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                return f"data:video/mp4;base64,{video_base64}"
            except Exception as e:
                import traceback
                print(f"ERROR: imageio/ffmpeg conversion failed: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                # Clean up temp file if it exists
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
                # Fall through to OpenCV method, but warn that it may not work
                print("WARNING: Falling back to OpenCV, but mp4v codec may not be browser-compatible")
                print("RECOMMENDATION: Install imageio-ffmpeg: pip install imageio-ffmpeg")
        
        # Fallback to OpenCV if imageio fails or not available
        if not CV2_AVAILABLE:
            return None
        
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Create video writer - try multiple codecs for browser compatibility
        codecs_to_try = ['XVID', 'mp4v', 'MJPG']
        out = None
        
        for codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                else:
                    if out:
                        out.release()
                    out = None
            except Exception as e:
                if out:
                    out.release()
                out = None
                continue
        
        if out is None or not out.isOpened():
            raise RuntimeError("Failed to initialize video writer with any codec")
        
        # Write frames
        for frame in video_array:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Read video file and encode to base64
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        # Create data URI
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        return f"data:video/mp4;base64,{video_base64}"
        
    except Exception as e:
        print(f"Error converting video to base64: {e}")
        import traceback
        traceback.print_exc()
        return None


def numpy_to_video_file(video_array: np.ndarray, output_path: str, fps: int = 30) -> bool:
    """
    Convert numpy video array to video file.
    
    Args:
        video_array: numpy array of shape (n_frames, height, width, channels) or (n_frames, height, width)
        output_path: path to save the video file
        fps: frames per second for the video
        
    Returns:
        True if successful, False otherwise
    """
    if not CV2_AVAILABLE:
        return False
    
    try:
        # Normalize video array to uint8
        if video_array.dtype != np.uint8:
            vmin, vmax = video_array.min(), video_array.max()
            if vmax > vmin:
                video_array = ((video_array - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                video_array = np.zeros_like(video_array, dtype=np.uint8)
        
        # Ensure correct shape
        if video_array.ndim == 3:
            video_array = video_array[..., np.newaxis]
            video_array = np.repeat(video_array, 3, axis=-1)
        
        n_frames, height, width, channels = video_array.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in video_array:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error converting video to file: {e}")
        return False


def get_video_frame_count(video_array: np.ndarray) -> int:
    """
    Get number of frames in video array.
    
    Args:
        video_array: numpy array of video frames
        
    Returns:
        Number of frames
    """
    if video_array is None:
        return 0
    if video_array.ndim >= 1:
        return video_array.shape[0]
    return 0





