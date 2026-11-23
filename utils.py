import os
import cv2
import numpy as np
import torchaudio
import subprocess
from pathlib import Path
from kornia.feature import LoFTR as KorniaLoFTR
import torch
torchaudio.set_audio_backend("soundfile")


# ================= VIDEO =================

def extract_frames_from_segment(video_path, output_dir, interval_sec=1, start_sec=0, duration_sec=5):
    """
    Extract frames directly from a time segment of the original video
    without saving an intermediate clip file.
    Uses ffmpeg to decode only the needed segment, then feeds frames to OpenCV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to pipe raw frames for just the needed segment
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(video_path),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Get frame dimensions from video
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        str(video_path)
    ]
    probe_out = subprocess.check_output(probe_cmd).decode().strip().split(",")
    width, height = int(probe_out[0]), int(probe_out[1])
    fps_num, fps_den = map(int, probe_out[2].split("/"))
    fps = fps_num / fps_den

    frame_size = width * height * 3
    step = max(1, int(fps * interval_sec))

    idx, saved = 0, 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if idx % step == 0:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            cv2.imwrite(str(output_dir / f"{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1

    proc.stdout.close()
    proc.wait()
    return saved


def generate_frame_pairs(frames_dir, pairs_file):
    frames = sorted(frames_dir.glob("*.jpg"))
    if len(frames) < 2:
        raise RuntimeError("Not enough frames for matching")

    with open(pairs_file, "w") as f:
        for i in range(len(frames) - 1):
            f.write(f"{frames[i].name} {frames[i+1].name}\n")


# ================= SUPERGLUE =================

def run_superglue(frames_dir, pairs_file, output_dir, superglue_script, force_cpu=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", superglue_script,
        "--input_dir", str(frames_dir),
        "--input_pairs", str(pairs_file),
        "--output_dir", str(output_dir),
        "--resize", "320", "240"
    ]

    subprocess.run(cmd, check=True)



def run_loftr(frames_dir, pairs_file, output_dir, weights="outdoor",
              resize=(320, 240), confidence_threshold=0.5, force_cpu=False):
    """

    Args:
        frames_dir           : Path to directory containing frame .jpg files
        pairs_file           : Path to text file, one "frame_a.jpg frame_b.jpg" per line
        output_dir           : Directory where output .npz files are saved
        weights              : LoFTR pretrained weights — "outdoor" or "indoor"
        resize               : (width, height) to resize frames before matching
        confidence_threshold : Discard matches with confidence below this value
        force_cpu            : Use CPU even when CUDA is available
    """
    

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Running LoFTR on device: {device}")
    matcher = KorniaLoFTR(pretrained=weights).to(device).eval()

    target_w, target_h = resize

    def _load_gray_tensor(img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.resize(img, (target_w, target_h))
        t = torch.from_numpy(img).float() / 255.0       # [H, W]
        return t.unsqueeze(0).unsqueeze(0).to(device)   # [1, 1, H, W]

    with open(pairs_file) as pf:
        pairs = [line.strip().split() for line in pf if line.strip()]

    with torch.inference_mode():
        for name0, name1 in pairs:
            img0_t = _load_gray_tensor(frames_dir / name0)
            img1_t = _load_gray_tensor(frames_dir / name1)

            batch = {"image0": img0_t, "image1": img1_t}
            results = matcher(batch)
            mkpts0 = results["keypoints0"].cpu().numpy()   # (M, 2)
            mkpts1 = results["keypoints1"].cpu().numpy()   # (M, 2)
            conf   = results["confidence"].cpu().numpy()   # (M,)

            mask   = conf >= confidence_threshold
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
            conf   = conf[mask]

            stem0    = Path(name0).stem
            stem1    = Path(name1).stem
            out_path = output_dir / f"{stem0}_{stem1}_matches_loftr.npz"

            np.savez(
                str(out_path),
                keypoints0=mkpts0.astype(np.float32),
                keypoints1=mkpts1.astype(np.float32),
                confidence=conf.astype(np.float32),
                matcher=np.array("loftr"),
            )

    del matcher
    if torch.cuda.is_available() and not force_cpu:
        torch.cuda.empty_cache()

def analyze_motion_superglue(npz_dir):
    """
    Analyze camera motion from SuperGlue .npz files.
    
    SuperGlue output:
        keypoints0  : (N, 2) — all detected keypoints in frame 0
        keypoints1  : (N, 2) — all detected keypoints in frame 1
        matches     : (N,)   — index into keypoints1, -1 = unmatched
    
    Convention:
        points flow RIGHT (mkpts1.x > mkpts0.x) → camera moved LEFT in world
        points flow LEFT  (mkpts1.x < mkpts0.x) → camera moved RIGHT in world
    """
    all_mkpts0 = []
    all_mkpts1 = []
    all_conf   = []

    for npz_path in npz_dir.glob("*matches.npz"):
        data    = np.load(npz_path)
        matches = data["matches"]           # (N,)
        k0      = data["keypoints0"]        # (N, 2)
        k1      = data["keypoints1"]        # (M, 2)
        scores  = data.get(
            "match_confidence",
            np.ones(len(matches), dtype=np.float32)
        )

        valid  = matches != -1
        mkpts0 = k0[valid]
        mkpts1 = k1[matches[valid]]
        conf   = scores[valid]

        if len(mkpts0) == 0:
            continue

        all_mkpts0.append(mkpts0)
        all_mkpts1.append(mkpts1)
        all_conf.append(conf)

    if not all_mkpts0:
        return None

    mkpts0 = np.concatenate(all_mkpts0, axis=0)
    mkpts1 = np.concatenate(all_mkpts1, axis=0)
    conf   = np.concatenate(all_conf,   axis=0)

    return _compute_motion_with_ransac(mkpts0, mkpts1, conf, matcher="superglue")


def analyze_motion_loftr(npz_dir, confidence_threshold=0.5):
    """
    Analyze motion from LoFTR .npz files.
    
    LoFTR output:
        keypoints0  : (N, 2) — matched points in frame 0
        keypoints1  : (N, 2) — matched points in frame 1
        confidence  : (N,)   — per-match confidence scores
    
    Every row is already a valid match — no index array, no -1 sentinel.
    
    Convention:
        points flow RIGHT (mkpts1.x > mkpts0.x) → camera moved LEFT in world
        points flow LEFT  (mkpts1.x < mkpts0.x) → camera moved RIGHT in world
        
        BUT for outliers (object motion):
        points flow RIGHT → object moved RIGHT in world  (same direction)
    """
    all_mkpts0 = []
    all_mkpts1 = []
    all_conf   = []

    for npz_path in npz_dir.glob("*loftr.npz"):
        data = np.load(npz_path, allow_pickle=True)

        if "matcher" not in data or str(data["matcher"]) != "loftr":
            continue

        mkpts0 = data["keypoints0"]   # (N, 2)
        mkpts1 = data["keypoints1"]   # (N, 2)
        conf   = data["confidence"]   # (N,)

        # Confidence filter — removes unreliable matches
        # before RANSAC to reduce noise
        mask   = conf >= confidence_threshold
        mkpts0 = mkpts0[mask]
        mkpts1 = mkpts1[mask]
        conf   = conf[mask]

        if len(mkpts0) == 0:
            continue

        all_mkpts0.append(mkpts0)
        all_mkpts1.append(mkpts1)
        all_conf.append(conf)

    if not all_mkpts0:
        return None

    mkpts0 = np.concatenate(all_mkpts0, axis=0)
    mkpts1 = np.concatenate(all_mkpts1, axis=0)
    conf   = np.concatenate(all_conf,   axis=0)

    return _compute_motion_with_ransac(mkpts0, mkpts1, conf, matcher="loftr")


def _compute_motion_with_ransac(mkpts0, mkpts1, conf, matcher):
    """
    Shared RANSAC-based motion analysis for both SuperGlue and LoFTR.
    
    Separates:
        Inliers  → background matches → camera motion
                   flow direction is OPPOSITE to camera world direction
                   
        Outliers → foreground matches → object motion  
                   flow direction is SAME as object world direction

    Returns dict with correct world-space direction labels for both.
    """
    if len(mkpts0) < 8:
        return None

    # --- RANSAC homography to separate background from foreground ---
    H, inlier_mask = cv2.findHomography(
        mkpts0, mkpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        confidence=0.999,
        maxIters=2000
    )

    if H is None or inlier_mask is None:
        return None

    inliers  = inlier_mask.ravel().astype(bool)
    outliers = ~inliers

    result = {
        "matcher":           matcher,
        "total_matches":     len(mkpts0),
        "num_inliers":       int(inliers.sum()),
        "num_outliers":      int(outliers.sum()),
        "inlier_ratio":      float(inliers.mean()),
        "mean_confidence":   float(conf.mean()),

        # Camera motion (from background inliers)
        "camera_world_side": None,
        "camera_net_flow":   None,   # raw point flow — positive = points moved right

        # Object motion (from foreground outliers)
        "object_world_side": None,
        "object_net_flow":   None,   # raw point flow — positive = points moved right
    }

    # --- Camera motion from background (inliers) ---
    # Point flow direction is OPPOSITE to camera world direction
    # points flow right (net_flow > 0) → camera moved LEFT
    # points flow left  (net_flow < 0) → camera moved RIGHT
    if inliers.sum() >= 10:
        dx_bg    = mkpts1[inliers, 0] - mkpts0[inliers, 0]
        net_flow = float(np.average(dx_bg, weights=conf[inliers]))

        result["camera_net_flow"]   = net_flow
        result["camera_world_side"] = "left" if net_flow > 0 else "right"

    # --- Object motion from foreground (outliers) ---
    # Point flow direction is SAME as object world direction
    # points flow right (net_flow > 0) → object moved RIGHT
    # points flow left  (net_flow < 0) → object moved LEFT
    if outliers.sum() >= 10:
        dx_obj   = mkpts1[outliers, 0] - mkpts0[outliers, 0]
        net_flow = float(np.average(dx_obj, weights=conf[outliers]))

        result["object_net_flow"]   = net_flow
        result["object_world_side"] = "right" if net_flow > 0 else "left"

    return result

# ================= AUDIO =================

def compute_audio_stats_segment(audio_path, start_sec=0, duration_sec=5, window_sec=1):
    """
    Compute audio spatial stats for a time segment.

    Returns:
        dict with keys:
            mean_iid_db      : float  — signed mean IID across active windows
            abs_mean_iid_db  : float  — absolute value, used for threshold filtering
            dominant_side    : "left" | "right" | "ambiguous"
            iid_consistency  : float  — fraction of windows agreeing with dominant side
            iid_per_second   : list   — per-window IID values (active windows only)
            num_active_windows: int   — windows with real audio (non-silent)
            num_seconds      : int    — total windows attempted
            temporal_iid_energy: float
    """
    import soundfile as sf
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        f.seek(int(start_sec * sr))
        data = f.read(int(duration_sec * sr), dtype="float32", always_2d=True)

    audio = data.T  # [channels, N]
    if audio.shape[0] != 2:
        raise RuntimeError("Audio must be stereo")

    left, right = audio[0], audio[1]
    samples_per_window = int(sr * window_sec)
    num_windows = min(
        len(left)  // samples_per_window,
        len(right) // samples_per_window
    )

    if num_windows < 3:
        raise RuntimeError("Audio segment too short")

    SILENCE_THRESHOLD = 1e-4  # RMS below this = silent window

    iid_series = []

    for i in range(num_windows):
        l_seg = left [i * samples_per_window:(i + 1) * samples_per_window]
        r_seg = right[i * samples_per_window:(i + 1) * samples_per_window]

        l_rms = np.sqrt(np.mean(l_seg ** 2))
        r_rms = np.sqrt(np.mean(r_seg ** 2))

        # Skip silent windows — don't inject 0.0 into the mean
        if l_rms < SILENCE_THRESHOLD or r_rms < SILENCE_THRESHOLD:
            continue

        iid = 20 * np.log10(r_rms / l_rms)
        iid_series.append(iid)

    # Not enough active windows to trust the label
    if len(iid_series) < 2:
        return {
            "mean_iid_db":         0.0,
            "abs_mean_iid_db":     0.0,
            "dominant_side":       "ambiguous",
            "iid_consistency":     0.0,
            "iid_per_second":      [],
            "num_active_windows":  len(iid_series),
            "num_seconds":         num_windows,
            "temporal_iid_energy": 0.0,
        }

    iid_series  = np.array(iid_series)
    mean_iid    = float(np.mean(iid_series))
    abs_mean_iid = abs(mean_iid)

    # Dead zone — don't assign a side to near-zero IID
    IID_DEAD_ZONE = 1.0  # dB
    if abs_mean_iid < IID_DEAD_ZONE:
        dominant_side = "ambiguous"
        iid_consistency = 0.0
    else:
        dominant_side = "right" if mean_iid > 0 else "left"
        expected_sign = 1 if dominant_side == "right" else -1
        iid_consistency = float(np.mean(np.sign(iid_series) == expected_sign))

    # Temporal IID energy (unchanged)
    temporal_iid_energy = 0.0
    for t in range(2, len(iid_series)):
        delta_prev = abs(iid_series[t - 1] - iid_series[t - 2])
        delta_curr = abs(iid_series[t]     - iid_series[t - 1])
        temporal_iid_energy += delta_prev * delta_curr

    return {
        "mean_iid_db":          mean_iid,
        "abs_mean_iid_db":      abs_mean_iid,
        "dominant_side":        dominant_side,
        "iid_consistency":      iid_consistency,
        "iid_per_second":       iid_series.tolist(),
        "num_active_windows":   len(iid_series),
        "num_seconds":          num_windows,
        "temporal_iid_energy":  float(temporal_iid_energy),
    }

def load_audio_segment(audio_path, start_sec, duration_sec):
    """
    Load a segment of a stereo audio file on-the-fly.
    Used by train.py's AudioVideoDataset instead of loading pre-saved clips.
    Returns: (waveform tensor [2, N], sample_rate)
    """
    # Load full file to get sr, then slice (efficient for moderate-size files;
    # for very large files consider using soundfile directly with seek)
    import soundfile as sf
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        frame_offset = int(start_sec * sr)
        num_frames = int(duration_sec * sr)
        f.seek(frame_offset)
        data = f.read(num_frames, dtype="float32", always_2d=True)  # shape: [N, channels]

    import torch
    waveform = torch.from_numpy(data.T)  # shape: [channels, N]

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    return waveform, sr