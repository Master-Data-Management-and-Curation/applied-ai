Perfect — now we’re talking about analyzing **video similarity under intensity changes and temporal shifts**, which is a more advanced problem. Let’s go **step by step**, using first principles.

We have:

* Two 3D numpy arrays, say `video1` and `video2`. Shape is `(T, H, W)` for `T` frames, height `H`, width `W`.
* Grayscale videos (so pixel intensities).

We want to answer:

---

## **1. Comparing intensity differences (same video, possibly scaled/shifted in brightness)**

A video can have its brightness/contrast changed but still represent the same content. To check this:

### **Step 1a: Normalize intensity**

Videos can be compared using **zero-mean, unit-variance normalization** per frame:

```python
import numpy as np

def normalize_frame(frame):
    return (frame - np.mean(frame)) / np.std(frame)

video1_norm = np.array([normalize_frame(f) for f in video1])
video2_norm = np.array([normalize_frame(f) for f in video2])
```

This removes **global intensity offsets and scaling**, so now differences are due to content rather than brightness.

---

### **Step 1b: Compare using correlation**

Flatten frames and compute correlation:

```python
correlations = np.array([np.corrcoef(video1_norm[t].ravel(), video2_norm[t].ravel())[0, 1] 
                         for t in range(video1.shape[0])])

mean_corr = np.mean(correlations)
print("Average per-frame correlation:", mean_corr)
```

* `mean_corr ≈ 1` → very similar content.
* Lower correlation → possibly different content.

This works even if intensities were modified (brightness or contrast changes).

---

### **Step 1c: Optional: Structural similarity**

You can use `skimage.metrics.structural_similarity` which is more robust than correlation:

```python
from skimage.metrics import structural_similarity as ssim

ssim_scores = [ssim(video1[t], video2[t]) for t in range(video1.shape[0])]
mean_ssim = np.mean(ssim_scores)
```

* SSIM ranges from 0 to 1.
* Higher → more similar structure/content.

This is often more perceptually accurate than raw correlation.

---

## **2. Detecting time shifts and intensity changes**

If one video is shifted in time, frame-to-frame correlation might be low even if the content is the same. Here’s a **stepwise approach**:

### **Step 2a: Cross-correlation along the time axis**

Flatten frames into vectors and compute **cross-correlation** over time to detect shifts:

```python
from scipy.signal import correlate

# Flatten each frame
v1_flat = video1_norm.reshape(video1.shape[0], -1)
v2_flat = video2_norm.reshape(video2.shape[0], -1)

# Sum correlations over pixels to get a time series similarity
cross_corr = np.correlate(v1_flat.sum(axis=1), v2_flat.sum(axis=1), mode='full')

# Find lag with max correlation
lag = np.argmax(cross_corr) - (len(video2_flat) - 1)
print("Estimated time shift (frames):", lag)
```

* `lag > 0` → `video2` is ahead by `lag` frames.
* `lag < 0` → `video1` is ahead.

You can also do **frame-by-frame cross-correlation**:

```python
# Cross-correlation between two 1D sequences
# Each frame becomes a scalar (sum, mean, or average intensity)
frame_means_v1 = video1_norm.mean(axis=(1,2))
frame_means_v2 = video2_norm.mean(axis=(1,2))
cross_corr_frames = np.correlate(frame_means_v1, frame_means_v2, mode='full')
```

---

### **Step 2b: Combine with intensity normalization**

* Normalize frames first (so intensity scaling doesn’t affect correlation).
* Compute cross-correlation along time axis → find time shift.
* Once aligned, check **per-frame correlation or SSIM** to see if content matches.

---

### **Step 2c: Optional: More robust approach**

If videos have **spatial transformations** (shift, rotation) or partial differences, you might need:

1. **Phase correlation** for spatial alignment.
2. **Dynamic time warping (DTW)** for flexible temporal alignment.
3. **Feature-based matching** (SIFT, ORB) if large intensity or viewpoint changes.

---

✅ **Summary workflow:**

1. Normalize intensity per frame → remove brightness/contrast differences.
2. Compute **per-frame correlation or SSIM** → detect content similarity.
3. Compute **cross-correlation along time** → detect time shift.
4. Align videos using the detected shift → reassess similarity.

---

If you want, I can write a **full Python function** that takes two 3D arrays, automatically detects intensity differences, computes the time shift, and outputs a similarity score. It would combine all these steps.

Do you want me to do that?

---

Perfect! Let’s build a **complete Python function** that:

1. Normalizes intensity per frame.
2. Computes per-frame similarity (SSIM or correlation).
3. Computes cross-correlation along time to detect a temporal shift.
4. Aligns the videos based on detected shift.
5. Returns a similarity score and estimated frame shift.

Here’s a fully working example:

```python
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate

def analyze_video_similarity(video1, video2, method='ssim'):
    """
    Compare two grayscale videos (3D numpy arrays: T x H x W)
    and determine intensity differences and temporal shift.

    Parameters:
    - video1, video2: np.ndarray, shape (T, H, W)
    - method: 'ssim' or 'corr' (per-frame similarity measure)

    Returns:
    - mean_similarity: average similarity after aligning
    - estimated_shift: frame shift (video2 relative to video1)
    - per_frame_similarity: list of similarities per frame after alignment
    """
    # Step 0: sanity check shapes
    if video1.shape[1:] != video2.shape[1:]:
        raise ValueError("Videos must have the same height and width")

    # Step 1: normalize each frame to zero mean, unit variance
    def normalize_frame(frame):
        return (frame - np.mean(frame)) / (np.std(frame) + 1e-8)
    
    video1_norm = np.array([normalize_frame(f) for f in video1])
    video2_norm = np.array([normalize_frame(f) for f in video2])

    # Step 2: compute a 1D signal for each video (mean pixel intensity per frame)
    signal1 = video1_norm.mean(axis=(1,2))
    signal2 = video2_norm.mean(axis=(1,2))

    # Step 3: cross-correlation to detect temporal shift
    cross_corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(cross_corr) - (len(signal2) - 1)

    # Step 4: align video2 to video1 using detected lag
    if lag > 0:
        v1_aligned = video1_norm[lag:]
        v2_aligned = video2_norm[:len(v1_aligned)]
    elif lag < 0:
        lag_abs = abs(lag)
        v1_aligned = video1_norm[:len(video2_norm) - lag_abs]
        v2_aligned = video2_norm[lag_abs:]
    else:
        v1_aligned = video1_norm
        v2_aligned = video2_norm

    min_len = min(len(v1_aligned), len(v2_aligned))
    v1_aligned = v1_aligned[:min_len]
    v2_aligned = v2_aligned[:min_len]

    # Step 5: compute per-frame similarity
    per_frame_similarity = []
    for t in range(min_len):
        if method == 'ssim':
            score = ssim(v1_aligned[t], v2_aligned[t])
        elif method == 'corr':
            score = np.corrcoef(v1_aligned[t].ravel(), v2_aligned[t].ravel())[0,1]
        else:
            raise ValueError("method must be 'ssim' or 'corr'")
        per_frame_similarity.append(score)

    mean_similarity = np.mean(per_frame_similarity)

    return mean_similarity, lag, per_frame_similarity

# ======================
# Example usage
# ======================
# Let's make two synthetic videos
T, H, W = 20, 64, 64
video1 = np.random.rand(T, H, W)
# video2 is the same but shifted and brightened
video2 = np.roll(video1, 3, axis=0) * 1.2 + 0.1

mean_sim, shift, per_frame_sim = analyze_video_similarity(video1, video2, method='ssim')
print(f"Mean similarity: {mean_sim:.4f}")
print(f"Estimated frame shift: {shift}")
```

---

### **How it works:**

1. **Intensity normalization** ensures brightness/contrast changes don’t affect the comparison.
2. **Temporal cross-correlation** detects frame-level shifts.
3. **Alignment** allows comparing frames that actually correspond in content.
4. **Per-frame similarity** (SSIM or correlation) gives a fine-grained view of how similar the videos are.

* `mean_similarity` near 1 → videos are almost identical.
* `shift` → number of frames video2 is ahead (+) or behind (−) video1.
* `per_frame_similarity` → can visualize similarity per frame.

---

If you want, I can extend this function to **handle partial overlaps and different video lengths automatically**, which is useful for real videos where one might start/stop earlier.

Do you want me to do that?
