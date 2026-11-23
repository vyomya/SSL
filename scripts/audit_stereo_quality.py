import soundfile as sf
from pathlib import Path
import numpy as np
from random import sample
import json
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import VIDEO_DIR, AUDIO_DIR, PROCESSED_DIR, SHORTLIST_DIR, CHECKPOINT_DIR, VGGSOUND_CSV

def verify_stereo_quality(audio_dir, manifest_path, n_samples=50):
    """
    Check actual stereo quality of audio files in your dataset.
    Verifies they are genuinely stereo (not just dual-mono).
    """
    audio_dir = Path(audio_dir)

    with open(manifest_path) as f:
        manifest = json.load(f)

    sampled = sample(manifest, min(n_samples, len(manifest)))

    truly_stereo    = 0
    dual_mono       = 0   # two channels but identical
    near_dual_mono  = 0   # two channels, very similar (correlation > 0.99)
    mono_files      = 0
    errors          = 0

    iid_values = []

    for entry in sampled:
        audio_path = Path(entry['audio_path'])
        start_sec  = entry['start_sec']

        try:
            with sf.SoundFile(str(audio_path)) as f:
                sr = f.samplerate
                f.seek(int(start_sec * sr))
                data = f.read(int(5 * sr), dtype="float32", always_2d=True)

            audio = data.T  # [channels, N]

            if audio.shape[0] == 1:
                mono_files += 1
                continue

            left  = audio[0]
            right = audio[1]

            # Check if channels are identical
            if np.allclose(left, right, atol=1e-6):
                dual_mono += 1
                continue

            # Check correlation
            corr = np.corrcoef(left, right)[0, 1]

            if corr > 0.99:
                near_dual_mono += 1
            else:
                truly_stereo += 1

            # Compute IID for this clip
            l_rms = np.sqrt(np.mean(left**2))
            r_rms = np.sqrt(np.mean(right**2))
            if l_rms > 1e-4 and r_rms > 1e-4:
                iid = 20 * np.log10(r_rms / l_rms)
                iid_values.append(abs(iid))

            print(f"  {audio_path.stem[:30]:30s} | "
                  f"corr={corr:.4f} | "
                  f"IID={20*np.log10(r_rms/l_rms) if l_rms>1e-4 and r_rms>1e-4 else 0:.3f} dB | "
                  f"L_rms={l_rms:.4f} R_rms={r_rms:.4f}")

        except Exception as e:
            errors += 1
            print(f"  ❌ Error: {audio_path.name}: {e}")

    total = len(sampled)
    print(f"\n{'='*60}")
    print(f"Sampled: {total} files")
    print(f"  Truly stereo (corr < 0.99):  {truly_stereo}  ({100*truly_stereo/total:.1f}%)")
    print(f"  Near dual-mono (corr>0.99):  {near_dual_mono} ({100*near_dual_mono/total:.1f}%)")
    print(f"  Dual-mono (identical):       {dual_mono}      ({100*dual_mono/total:.1f}%)")
    print(f"  Mono files:                  {mono_files}     ({100*mono_files/total:.1f}%)")
    print(f"  Errors:                      {errors}")

    if iid_values:
        iid_values = np.array(iid_values)
        print(f"\nIID distribution (|mean_iid_db|) across active clips:")
        print(f"  Min:      {iid_values.min():.4f} dB")
        print(f"  25th pct: {np.percentile(iid_values, 25):.4f} dB")
        print(f"  Median:   {np.percentile(iid_values, 50):.4f} dB")
        print(f"  75th pct: {np.percentile(iid_values, 75):.4f} dB")
        print(f"  90th pct: {np.percentile(iid_values, 90):.4f} dB")
        print(f"  Max:      {iid_values.max():.4f} dB")
        print(f"\n  Clips with |IID| > 1.0 dB: "
              f"{(iid_values > 1.0).sum()} ({100*(iid_values>1.0).mean():.1f}%)")
        print(f"  Clips with |IID| > 0.5 dB: "
              f"{(iid_values > 0.5).sum()} ({100*(iid_values>0.5).mean():.1f}%)")
        print(f"  Clips with |IID| > 0.1 dB: "
              f"{(iid_values > 0.1).sum()} ({100*(iid_values>0.1).mean():.1f}%)")

verify_stereo_quality(
    audio_dir     = AUDIO_DIR,
    manifest_path = SHORTLIST_DIR / "index.json",
    n_samples     = 50
)