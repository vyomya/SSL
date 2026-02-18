import os
import cv2
import numpy as np
import torchaudio
import subprocess
from pathlib import Path

torchaudio.set_audio_backend("soundfile")
import subprocess
from pathlib import Path


def split_video_audio(
    video_path: Path,
    audio_path: Path,
    output_dir: Path,
    clip_sec: int = 5,
    stride: int = 3
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    duration = float(subprocess.check_output(cmd).decode().strip())

    clips = []
    start = 0
    while start + clip_sec <= duration:
        end = start + clip_sec
        clip_name = f"clip_{start:04d}_{end:04d}"
        clip_dir = output_dir / clip_name
        clip_dir.mkdir(exist_ok=True)

        video_out = clip_dir / "video.mp4"
        audio_out = clip_dir / "audio.wav"

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(clip_sec),
            "-i", str(video_path),
            "-c:v", "libx264", "-an",
            str(video_out)
        ], check=True)

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-t", str(clip_sec),
            "-i", str(audio_path),
            "-ac", "2",
            str(audio_out)
        ], check=True)

        clips.append(clip_dir)
        start += stride

    return clips


# ================= VIDEO =================

def extract_frames(video_path, output_dir, interval_sec=1):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps * interval_sec))

    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            cv2.imwrite(str(output_dir / f"{saved:05d}.jpg"), frame)
            saved += 1
        idx += 1

    cap.release()
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

    if force_cpu:
        cmd.append("--force_cpu")

    subprocess.run(cmd, check=True)

def run_loftr(frames_dir, pairs_file, output_dir, loftr_script, force_cpu=False):
    
    pass

# ================= MOTION =================

def analyze_motion(npz_dir):
    movement = {"left": 0, "right": 0, "up": 0, "down": 0}

    for npz in npz_dir.glob("*.npz"):
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

def compute_audio_stats(audio_path, window_sec=1):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.numpy()

    if audio.shape[0] != 2:
        raise RuntimeError("Audio must be stereo")

    left, right = audio[0], audio[1]
    samples_per_window = int(sr * window_sec)

    num_windows = min(
        len(left) // samples_per_window,
        len(right) // samples_per_window
    )

    if num_windows < 3:
        raise RuntimeError("Audio too short for temporal IID computation")

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

