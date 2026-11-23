import os
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent
DATA_ROOT      = Path(os.environ.get("SSL_DATA_ROOT", "/fs/nexus-scratch/vyomwal5/vgg"))
VIDEO_DIR      = DATA_ROOT / "video"
AUDIO_DIR      = DATA_ROOT / "audio"
PROCESSED_DIR  = DATA_ROOT / "processed"
SHORTLIST_DIR  = DATA_ROOT / "shortlisted"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
VGGSOUND_CSV   = REPO_ROOT / "data" / "vggsound.csv"
SUPERGLUE_SCRIPT = REPO_ROOT / "third_party" / "SuperGluePretrainedNetwork" / "match_pairs.py"