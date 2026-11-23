import csv
import subprocess
import os

BASE_DIR = "audioset_data"
VIDEO_DIR = os.path.join(BASE_DIR, "video")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

def download_video(ytid, start, end):
    """Download video segment only."""
    url = f"https://www.youtube.com/watch?v={ytid}"
    output_template = os.path.join(VIDEO_DIR, f"{ytid}_{start}_{end}.mp4")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", output_template,
        url
    ]
    subprocess.run(cmd, check=False)

    return output_template

def extract_audio(video_path):
    """Extract audio from video using ffmpeg."""
    filename = os.path.basename(video_path).replace(".mp4", ".wav")
    audio_path = os.path.join(AUDIO_DIR, filename)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",      # no video
        "-acodec", "pcm_s16le",  # WAV
        audio_path
    ]
    subprocess.run(cmd, check=False)

    return audio_path


def process_segment(ytid, start, end):
    video_path = download_video(ytid, start, end)
    extract_audio(video_path)


# --------------------------- RUN ON CSV ----------------------------
CSV_PATH = "vggsound.csv"

with open(CSV_PATH) as f:
    reader = csv.reader(f)
    for i,row in enumerate(reader):
        if row[0].startswith("#"):
            continue
        ytid, start, end = row[0], row[1], int(row[1])+10
        if i==100:
            break
        process_segment(ytid, start, end)
