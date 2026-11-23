import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from torchvision import models
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import time
from collections import defaultdict
import random
from utils import load_audio_segment
from config import SHORTLIST_DIR, PROCESSED_DIR, CHECKPOINT_DIR
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioVideoDataset(Dataset):
    """
    Dataset for temporal audio pairs.
    Labels are derived from the corrected world-space motion format
    (camera_world_side, object_world_side) produced by _compute_motion_with_ransac.
    """

    def __init__(self, manifest_path, processed_base_dir,
                 sample_rate=48000, clip_duration=2.5):
        self.processed_base_dir = Path(processed_base_dir)
        self.sample_rate        = sample_rate
        self.clip_duration      = clip_duration
        self.clip_length        = int(sample_rate * clip_duration)

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def load_json_data(self, video_id, clip_id):
        json_path = self.processed_base_dir / video_id / clip_id / "result.json"
        if not json_path.exists():
            raise FileNotFoundError(f"result.json not found: {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)

    def __getitem__(self, idx):
        entry      = self.manifest[idx]
        video_id   = entry['video_id']
        clip_id    = entry['clip_id']
        audio_path = entry['audio_path']
        start_sec  = entry['start_sec']

        json_data = self.load_json_data(video_id, clip_id)

        # ── Load and split audio ──────────────────────────────────────────
        waveform, sr = load_audio_segment(audio_path, start_sec, self.clip_duration * 2)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        mid              = waveform.shape[1] // 2
        waveform_t       = self._pad_or_trim(waveform[:, :mid])
        waveform_t_prime = self._pad_or_trim(waveform[:, mid:mid * 2])

        spectrogram_t       = self.waveform_to_spectrogram(waveform_t)
        spectrogram_t_prime = self.waveform_to_spectrogram(waveform_t_prime)

        # ── Labels from both motion cues ──────────────────────────────────
        labels_sp    = self.extract_labels_from_json(
            json_data, motion_key='visual_motion')
        labels_loftr = self.extract_labels_from_json(
            json_data, motion_key='visual_motion_loftr')

        # ── IID features ──────────────────────────────────────────────────
        iid = self.extract_iid_features(json_data)

        return {
            'spectrogram_t':       spectrogram_t,
            'spectrogram_t_prime': spectrogram_t_prime,

            # SuperGlue labels
            'sp_rotation_direction':    labels_sp['rotation_direction'],
            'sp_translation_direction': labels_sp['translation_direction'],
            'sp_rotation_class':        labels_sp['rotation_class'],
            'sp_translation_class':     labels_sp['translation_class'],

            # LoFTR labels
            'loftr_rotation_direction':    labels_loftr['rotation_direction'],
            'loftr_translation_direction': labels_loftr['translation_direction'],
            'loftr_rotation_class':        labels_loftr['rotation_class'],
            'loftr_translation_class':     labels_loftr['translation_class'],

            # Shared IID
            'iid_energy':    iid['temporal_iid_energy'],
            'mean_iid_db':   iid['mean_iid_db'],
            'dominant_side': iid['dominant_side_encoded'],

            'video_id': video_id,
            'clip_id':  clip_id,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pad_or_trim(self, w):
        if w.shape[1] < self.clip_length:
            return torch.nn.functional.pad(w, (0, self.clip_length - w.shape[1]))
        return w[:, :self.clip_length]

    def waveform_to_spectrogram(self, waveform):
        n_fft      = 512
        hop_length = 256
        spectrograms = []
        for ch in range(2):
            stft = torch.stft(
                waveform[ch], n_fft=n_fft, hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(n_fft)
            )
            mag   = torch.log1p(torch.abs(stft))
            phase = torch.angle(stft)
            for arr in (mag, phase):
                arr = torch.nn.functional.interpolate(
                    arr.unsqueeze(0).unsqueeze(0),
                    size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze()
                spectrograms.append(arr)
        spec = torch.stack(spectrograms, dim=0)
        for i in range(spec.shape[0]):
            std = spec[i].std()
            if std > 0:
                spec[i] = (spec[i] - spec[i].mean()) / std
        return spec

    def extract_labels_from_json(self, json_data, motion_key='visual_motion',
                                  n_classes=32):
        """
        Derive rotation/translation labels from world-space motion direction.

        Uses camera_world_side as primary signal (inlier-backed, reliable).
        Falls back to object_world_side if camera motion is absent.
        Maps directional side to a class index:
            left  → class 0  (leftward motion bin)
            right → class 1  (rightward motion bin)
        Magnitude class derived from net_flow magnitude.
        """
        vm = json_data.get(motion_key, {})
        if not vm:
            return {
                'rotation_direction':    -100,  # neutral
                'translation_direction': -100,  # neutral
                'rotation_class':        0,
                'translation_class':     0,
            }
        # ── Determine dominant motion side ────────────────────────────────
        camera_side     = vm.get('camera_world_side')
        object_side     = vm.get('object_world_side')
        camera_net_flow = vm.get('camera_net_flow', 0.0) or 0.0
        object_net_flow = vm.get('object_net_flow', 0.0) or 0.0
        inlier_ratio    = vm.get('inlier_ratio', 0.0)    or 0.0

        MOTION_THRESHOLD = 1.0  # pixels

        has_camera = (camera_side is not None and
                      abs(camera_net_flow) > MOTION_THRESHOLD)
        has_object = (object_side is not None and
                      abs(object_net_flow) > MOTION_THRESHOLD)

        # Pick the dominant side using same priority as passes_shortlist_rule
        if has_camera and (not has_object or inlier_ratio >= 0.75):
            dominant_side = camera_side
            net_flow      = camera_net_flow
        elif has_object:
            dominant_side = object_side
            net_flow      = object_net_flow
        else:
            # No significant motion — default to center/neutral
            dominant_side = None
            net_flow      = 0.0

        # ── Direction class ───────────────────────────────────────────────
        # Map left/right to a class spread across n_classes bins
        # left  → bins [0,  n_classes//2)
        # right → bins [n_classes//2, n_classes)
        if dominant_side == 'right':
            base_class = n_classes // 2
        elif dominant_side == 'left':
            base_class = 0
        else:
            return {'rotation_direction': -100, 'translation_direction': -100,
                    'rotation_class': 0, 'translation_class': 0}

        # Use flow magnitude to spread within the half
        # Stronger flow → higher bin within the half
        MAX_FLOW     = 50.0  # pixels — clamp for stability
        flow_ratio   = min(abs(net_flow) / MAX_FLOW, 1.0)
        half         = n_classes // 2
        offset       = int(flow_ratio * (half - 1))
        direction_class = (base_class + offset) % n_classes

        # ── Magnitude class ───────────────────────────────────────────────
        magnitude_class = min(int(abs(net_flow) / (MAX_FLOW / n_classes)),
                              n_classes - 1)

        return {
            'rotation_direction':    direction_class,
            'translation_direction': direction_class,
            'rotation_class':        magnitude_class,
            'translation_class':     magnitude_class,
        }

    def extract_iid_features(self, json_data):
        """
        Extract IID features using updated audio stats format.
        Handles both old format (missing abs_mean_iid_db) and new format.
        """
        sp = json_data.get('audio_spatial', {})

        # Support both old and new format
        mean_iid    = sp.get('mean_iid_db', 0.0) or 0.0
        abs_mean_iid = sp.get('abs_mean_iid_db', abs(mean_iid))

        dominant_raw = sp.get('dominant_side', 'ambiguous')
        side_enc = {
            'left':      0,
            'right':     2,
            'ambiguous': 1,  # treat ambiguous as center
            'center':    1,
        }
        dominant_encoded = side_enc.get(dominant_raw, 1)

        return {
            'temporal_iid_energy':   float(sp.get('temporal_iid_energy', 0.0) or 0.0),
            'mean_iid_db':           float(mean_iid),
            'abs_mean_iid_db':       float(abs_mean_iid),
            'iid_consistency':       float(sp.get('iid_consistency', 0.0) or 0.0),
            'dominant_side_encoded': dominant_encoded,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpatialAudioEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.resnet    = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim  = self.resnet.fc.in_features
        self.resnet.fc    = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class SpatialSoundLocalizationModel(nn.Module):
    def __init__(self, n_classes=32, in_channels=4):
        super().__init__()
        self.audio_encoder = SpatialAudioEncoder(in_channels=in_channels)
        d = self.audio_encoder.feature_dim

        # IID input expanded to include abs_mean_iid and consistency
        self.localization_head = nn.Sequential(
            nn.Linear(d + 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        self.relative_rotation_head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        self.relative_translation_head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        self.rotation_magnitude_head    = nn.Linear(d * 2, n_classes)
        self.translation_magnitude_head = nn.Linear(d * 2, n_classes)

        # 3 classes: left / ambiguous / right
        self.dominant_side_head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, spec_t, spec_t_prime,
                iid_energy=None, mean_iid_db=None,
                abs_mean_iid=None, iid_consistency=None):
        f_t  = self.audio_encoder(spec_t)
        f_tp = self.audio_encoder(spec_t_prime)

        # Build IID feature vector — 4 dims
        bs = f_t.size(0)
        def _feat(x):
            if x is not None:
                return x.float().unsqueeze(1) if x.dim() == 1 else x.float()
            return torch.zeros(bs, 1, device=f_t.device)

        iid_vec = torch.cat([
            _feat(iid_energy),
            _feat(mean_iid_db),
            _feat(abs_mean_iid),
            _feat(iid_consistency),
        ], dim=1)  # [B, 4]

        loc_t  = self.localization_head(torch.cat([f_t,  iid_vec], dim=1))
        loc_tp = self.localization_head(torch.cat([f_tp, iid_vec], dim=1))
        combined = torch.cat([f_t, f_tp], dim=1)

        return {
            'localization_t':        loc_t,
            'localization_t_prime':  loc_tp,
            'relative_rotation':     self.relative_rotation_head(combined),
            'relative_translation':  self.relative_translation_head(combined),
            'rotation_magnitude':    self.rotation_magnitude_head(combined),
            'translation_magnitude': self.translation_magnitude_head(combined),
            'dominant_side':         self.dominant_side_head(f_t),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def combined_loss(outputs, targets, lambda1=0.9, lambda2=1.0):
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    L_rot   = ce(outputs['relative_rotation'],    targets['rotation_direction'])
    L_trans = ce(outputs['relative_translation'],  targets['translation_direction'])
    L_side  = ce(outputs['dominant_side'],         targets['dominant_side'])
    total   = lambda1 * L_rot + (1 - lambda1) * L_trans + lambda2 * L_side
    return total, {
        'rotation':      L_rot.item(),
        'translation':   L_trans.item(),
        'dominant_side': L_side.item(),
        'total':         total.item(),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_targets(batch, prefix, device):
    return {
        'rotation_direction':    batch[f'{prefix}rotation_direction'].to(device),
        'translation_direction': batch[f'{prefix}translation_direction'].to(device),
        'rotation_class':        batch[f'{prefix}rotation_class'].to(device),
        'translation_class':     batch[f'{prefix}translation_class'].to(device),
        'dominant_side':         batch['dominant_side'].to(device),
    }


def _run_epoch(models_dict, loaders_dict, optimizers_dict, device,
               lambda1, lambda2, train=True):
    label_prefix = {'sp': 'sp_', 'loftr': 'loftr_'}

    stats = {
        name: {
            'losses':  {k: 0.0 for k in
                        ('rotation', 'translation', 'dominant_side', 'total')},
            'correct': 0,
            'total':   0,
            'n_batches': 0,  # ← track per-model batch count independently
        }
        for name in models_dict
    }

    for name, model in models_dict.items():
        model.train() if train else model.eval()

    for name, model in models_dict.items():
        loader = loaders_dict[name]

        for batch in loader:
            spec_t        = batch['spectrogram_t'].to(device)
            spec_t_prime  = batch['spectrogram_t_prime'].to(device)
            iid_energy    = batch['iid_energy'].to(device).float()
            mean_iid_db   = batch['mean_iid_db'].to(device).float()
            # New IID features
            abs_mean_iid  = batch.get('abs_mean_iid_db')
            iid_consistency = batch.get('iid_consistency')
            if abs_mean_iid  is not None:
                abs_mean_iid   = abs_mean_iid.to(device).float()
            if iid_consistency is not None:
                iid_consistency = iid_consistency.to(device).float()

            targets = _make_targets(batch, label_prefix[name], device)

            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                outputs = model(
                    spec_t, spec_t_prime,
                    iid_energy, mean_iid_db,
                    abs_mean_iid, iid_consistency
                )
                loss, loss_dict = combined_loss(
                    outputs, targets, lambda1, lambda2)

            if train:
                optimizers_dict[name].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizers_dict[name].step()

            for k in stats[name]['losses']:
                stats[name]['losses'][k] += loss_dict[k]

            _, pred = torch.max(outputs['relative_rotation'], 1)
            tgt   = targets['rotation_direction']
            valid = tgt != -100
            stats[name]['total']   += valid.sum().item()
            stats[name]['correct'] += (pred[valid] == tgt[valid]).sum().item()

        # ── Average losses over this model's own batches ──────────────────
        n = max(stats[name]['n_batches'], 1)
        for k in stats[name]['losses']:
            stats[name]['losses'][k] /= n
        stats[name]['acc'] = (100.0 * stats[name]['correct'] /
                              max(stats[name]['total'], 1))

    return stats


# ---------------------------------------------------------------------------
# Video-level train/val split
# ---------------------------------------------------------------------------

def split_by_video(dataset, val_videos):
    video_to_indices = defaultdict(list)
    for i, entry in enumerate(dataset.manifest):
        video_to_indices[entry['video_id']].append(i)
    train_videos = set(video_to_indices) - val_videos
    val_videos   = set(video_to_indices) & val_videos

    train_indices = [i for vid in train_videos for i in video_to_indices[vid]]
    val_indices   = [i for vid in val_videos   for i in video_to_indices[vid]]

    print(f"  Train: {len(train_videos)} videos, {len(train_indices)} clips")
    print(f"  Val:   {len(val_videos)}   videos, {len(val_indices)}   clips")
    print(f"  Overlap check: {len(train_videos & val_videos)} shared videos (must be 0)")

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset   = torch.utils.data.Subset(dataset, val_indices)
    return train_subset, val_subset


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_dual_models(model_sp, model_loftr,
                      train_loader, val_loader,
                      train_loftr_loader, val_loftr_loader,
                      num_epochs=100, lambda1=0.9, lambda2=1.0,
                      device='cuda', save_dir='.'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models_dict        = {'sp': model_sp.to(device), 'loftr': model_loftr.to(device)}
    train_loaders_dict = {'sp': train_loader,       'loftr': train_loftr_loader}
    val_loaders_dict   = {'sp': val_loader,         'loftr': val_loftr_loader}

    optimizers_dict = {
        name: optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
        for name, m in models_dict.items()
    }
    schedulers_dict = {
        name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for name, opt in optimizers_dict.items()
    }

    best_val_acc   = {'sp': 0.0, 'loftr': 0.0}
    last_val_acc   = {'sp': 0.0, 'loftr': 0.0}
    last_train_acc = {'sp': 0.0, 'loftr': 0.0}

    for epoch in range(num_epochs):
        print(f'\n{"="*70}')
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'{"="*70}')
        t0 = time.time()

        train_stats = _run_epoch(
            models_dict, train_loaders_dict, optimizers_dict,
            device, lambda1, lambda2, train=True)
        val_stats = _run_epoch(
            models_dict, val_loaders_dict, optimizers_dict,
            device, lambda1, lambda2, train=False)

        print(f'Epoch time: {time.time() - t0:.2f} seconds')

        for name in ('sp', 'loftr'):
            tag = 'SuperPoint/SuperGlue' if name == 'sp' else 'LoFTR'
            tl  = train_stats[name]['losses']
            vl  = val_stats[name]['losses']
            ta  = train_stats[name]['acc']
            va  = val_stats[name]['acc']
            last_val_acc[name]   = va
            last_train_acc[name] = ta

            print(f'\n  [{tag}]')
            print(f'  Train  Acc: {ta:.2f}% | Total: {tl["total"]:.4f} | '
                  f'Rot: {tl["rotation"]:.4f} | Trans: {tl["translation"]:.4f} | '
                  f'Side: {tl["dominant_side"]:.4f}')
            print(f'  Val    Acc: {va:.2f}% | Total: {vl["total"]:.4f} | '
                  f'Rot: {vl["rotation"]:.4f} | Trans: {vl["translation"]:.4f} | '
                  f'Side: {vl["dominant_side"]:.4f}')

            if va > best_val_acc[name]:
                best_val_acc[name] = va
                torch.save({
                    'epoch':                epoch,
                    'motion_cue':           name,
                    'model_state_dict':     models_dict[name].state_dict(),
                    'optimizer_state_dict': optimizers_dict[name].state_dict(),
                    'val_acc':              va,
                    'train_acc':            ta,
                    'lambda1': lambda1, 'lambda2': lambda2,
                }, save_dir / f'best_{name}_model.pth')
                print("###Updated")
            print(f'  *** Best {tag} model saved (val_acc: {best_val_acc[name]:.2f}%) ***')
    
        for sched in schedulers_dict.values():
            sched.step()

    # ── Final checkpoints ─────────────────────────────────────────────────
    print(f'\n{"="*70}')
    for name in ('sp', 'loftr'):
        tag  = 'SuperPoint/SuperGlue' if name == 'sp' else 'LoFTR'
        path = save_dir / f'final_{name}_model.pth'
        torch.save({
            'epoch':                num_epochs - 1,
            'motion_cue':           name,
            'model_state_dict':     models_dict[name].state_dict(),
            'optimizer_state_dict': optimizers_dict[name].state_dict(),
            'val_acc':              last_val_acc[name],
            'train_acc':            last_train_acc[name],
            'lambda1': lambda1, 'lambda2': lambda2,
        }, path)
        print(f'[{tag}] final → {path}')
        print(f'[{tag}] best  → {save_dir}/best_{name}_model.pth  '
              f'(val_acc: {best_val_acc[name]:.2f}%)')

    return models_dict['sp'], models_dict['loftr']


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_classes  = 32
    batch_size = 16
    num_epochs = 100
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    lambda1, lambda2 = 0.9, 1.0

    MANIFEST_PATH       = SHORTLIST_DIR / "index.json"
    MANIFEST_LOFTR_PATH = SHORTLIST_DIR / "index_loftr.json"
    SAVE_DIR            = CHECKPOINT_DIR
    LOAD_PRETRAINED     = None  # set to SAVE_DIR to resume

    # ── Datasets with video-level splits ─────────────────────────────────
    all_vids = sorted({e['video_id'] for e in full_dataset_sp.manifest} |
                  {e['video_id'] for e in full_dataset_loftr.manifest})
    random.Random(42).shuffle(all_vids)
    val_videos = set(all_vids[:max(1, int(0.2 * len(all_vids)))])
    train_dataset_sp,    val_dataset_sp    = split_by_video(full_dataset_sp, val_videos)
    train_dataset_loftr, val_dataset_loftr = split_by_video(full_dataset_loftr, val_videos)

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset_sp, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(
        val_dataset_sp, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    train_loftr_loader = DataLoader(
        train_dataset_loftr, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        generator=torch.Generator().manual_seed(42))
    val_loftr_loader = DataLoader(
        val_dataset_loftr, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # ── Models ────────────────────────────────────────────────────────────
    model_sp    = SpatialSoundLocalizationModel(n_classes=n_classes, in_channels=4)
    model_loftr = SpatialSoundLocalizationModel(n_classes=n_classes, in_channels=4)

    if LOAD_PRETRAINED:
        for name, model in [('sp', model_sp), ('loftr', model_loftr)]:
            ckpt_path = LOAD_PRETRAINED / f'best_{name}_model.pth'
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded {name} checkpoint from {ckpt_path} "
                      f"(val_acc: {ckpt.get('val_acc', '?'):.2f}%)")

    print(f"\nParameters per model: {sum(p.numel() for p in model_sp.parameters()):,}")
    print(f"Device: {device}")
    print(f"  model_sp    ← visual_motion       (SuperPoint/SuperGlue)")
    print(f"  model_loftr ← visual_motion_loftr (LoFTR)")
    print(f"Checkpoints → {SAVE_DIR}\n")

    train_dual_models(
        model_sp, model_loftr,
        train_loader, val_loader,
        train_loftr_loader, val_loftr_loader,
        num_epochs=num_epochs,
        lambda1=lambda1, lambda2=lambda2,
        device=device,
        save_dir=SAVE_DIR,
    )