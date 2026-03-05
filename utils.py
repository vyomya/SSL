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



# ================= MOTION =================


def analyze_motion_loftr(npz_dir, confidence_threshold=0.0):
    """
    Analyze motion from LoFTR .npz output files.
    Each .npz must contain:
        keypoints0  : (N, 2) float32 — matched points in frame 0
        keypoints1  : (N, 2) float32 — matched points in frame 1
        confidence  : (N,)   float32 — per-match confidence scores
        matcher     : str            — must equal "loftr"

    Unlike SuperGlue output, every row is already a valid match —
    there is no index array and no -1 sentinel.

    Args:
        npz_dir              : directory containing LoFTR .npz files
        confidence_threshold : optional additional confidence filter applied
                               at analysis time (on top of the threshold used
                               during run_loftr). Useful for sweeping thresholds
                               without re-running the matcher.

    Returns:
        dict with keys "left", "right", "up", "down" — match-weighted
        directional motion counts, plus matcher metadata.
    """
    movement = {
        "left": 0, "right": 0,
        "up": 0,   "down": 0,
        "total_matches": 0,
        "mean_confidence": 0.0,
        "matcher": "loftr",
    }

    all_conf = []

    for npz_path in npz_dir.glob("*loftr.npz"):
        data = np.load(npz_path, allow_pickle=True)

        # Only process LoFTR files
        if "matcher" not in data or str(data["matcher"]) != "loftr":
            continue

        mkpts0 = data["keypoints0"]   # (N, 2)
        mkpts1 = data["keypoints1"]   # (N, 2)
        conf   = data["confidence"]   # (N,)

        # Optional additional confidence filter at analysis time
        if confidence_threshold > 0.0:
            mask   = conf >= confidence_threshold
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]
            conf   = conf[mask]

        if len(mkpts0) == 0:
            continue

        all_conf.append(conf)

        # Flow vector: delta = kpt1 - kpt0
        # Positive dx  → point moved right in frame  → camera panned left
        # Negative dx  → point moved left in frame   → camera panned right
        # We accumulate raw pixel-flow direction counts,
        # optionally weighted by confidence for richer signal.
        dx = mkpts1[:, 0] - mkpts0[:, 0]
        dy = mkpts1[:, 1] - mkpts0[:, 1]

        # Confidence-weighted counts
        movement["right"] += int(np.round(np.sum(conf[dx > 0])))
        movement["left"]  += int(np.round(np.sum(conf[dx < 0])))
        movement["down"]  += int(np.round(np.sum(conf[dy > 0])))
        movement["up"]    += int(np.round(np.sum(conf[dy < 0])))
        movement["total_matches"] += len(mkpts0)

    if all_conf:
        movement["mean_confidence"] = float(np.mean(np.concatenate(all_conf)))

    return movement

def analyze_motion(npz_dir):
    movement = {"left": 0, "right": 0, "up": 0, "down": 0}

    for npz in npz_dir.glob("*matches.npz"):
        data = np.load(npz)
        matches = data["matches"]
        k0 = data["keypoints0"]
        k1 = data["keypoints1"]

        for i, m in enumerate(matches):
            if m == -1:
                continue

            p0, p1 = k0[i], k1[m]
            movement["right" if p0[0] > p1[0] else "left"] += 1
            movement["down" if p0[1] > p1[1] else "up"] += 1

    return movement


# ================= AUDIO =================

def compute_audio_stats_segment(audio_path, start_sec=0, duration_sec=5, window_sec=1):
    """
    Compute audio spatial stats for a time segment of the original audio file
    without saving an intermediate clip.
    """
    import soundfile as sf
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        f.seek(int(start_sec * sr))
        data = f.read(int(duration_sec * sr), dtype="float32", always_2d=True)  # [N, channels]

    audio = data.T  # [channels, N]

    if audio.shape[0] != 2:
        raise RuntimeError("Audio must be stereo")

    left, right = audio[0], audio[1]
    samples_per_window = int(sr * window_sec)

    num_windows = min(
        len(left) // samples_per_window,
        len(right) // samples_per_window
    )

    if num_windows < 3:
        raise RuntimeError("Audio segment too short for temporal IID computation")

    # ---------- Per-second IID ----------
    iid_series = []

    for i in range(num_windows):
        l_seg = left[i * samples_per_window:(i + 1) * samples_per_window]
        r_seg = right[i * samples_per_window:(i + 1) * samples_per_window]

        l_rms = np.sqrt(np.mean(l_seg ** 2))
        r_rms = np.sqrt(np.mean(r_seg ** 2))

        if l_rms > 0 and r_rms > 0:
            iid = 20 * np.log10(r_rms / l_rms)
        else:
            iid = 0.0

        iid_series.append(iid)

    iid_series = np.array(iid_series)

    # ---------- Absolute product of IID changes ----------
    temporal_iid_energy = 0.0

    for t in range(2, len(iid_series)):
        delta_prev = abs(iid_series[t - 1] - iid_series[t - 2])
        delta_curr = abs(iid_series[t] - iid_series[t - 1])
        temporal_iid_energy += delta_prev * delta_curr

    # ---------- Dominant side ----------
    mean_iid = np.mean(iid_series)
    dominant_side = "right" if mean_iid > 0 else "left"

    return {
        "temporal_iid_energy": float(temporal_iid_energy),
        "mean_iid_db": float(mean_iid),
        "dominant_side": dominant_side,
        "iid_per_second": iid_series.tolist(),
        "num_seconds": int(num_windows)
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