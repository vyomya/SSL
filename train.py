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

class AudioVideoDataset(Dataset):
    """
    Dataset for temporal audio pairs.
    Each item returns labels computed from BOTH visual_motion and
    visual_motion_loftr so the two models can be trained concurrently
    from a single DataLoader pass.
    """

    def __init__(self, manifest_path, processed_base_dir, sample_rate=48000, clip_duration=2.5):
        self.processed_base_dir = Path(processed_base_dir)
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_length = int(sample_rate * clip_duration)

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
        entry = self.manifest[idx]
        video_id   = entry['video_id']
        clip_id    = entry['clip_id']
        audio_path = entry['audio_path']
        start_sec  = entry['start_sec']
        end_sec    = entry['end_sec']

        json_data = self.load_json_data(video_id, clip_id)

        # ---- load audio ----
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        start_frame = int(start_sec * self.sample_rate)
        end_frame   = int(end_sec   * self.sample_rate)
        waveform = waveform[:, start_frame:end_frame]

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # ---- split into t and t' ----
        mid = waveform.shape[1] // 2
        waveform_t       = waveform[:, :mid]
        waveform_t_prime = waveform[:, mid:mid * 2]

        for attr in ('waveform_t', 'waveform_t_prime'):
            w = locals()[attr]
            if w.shape[1] < self.clip_length:
                w = torch.nn.functional.pad(w, (0, self.clip_length - w.shape[1]))
            else:
                w = w[:, :self.clip_length]
            if attr == 'waveform_t':
                waveform_t = w
            else:
                waveform_t_prime = w

        spectrogram_t       = self.waveform_to_spectrogram(waveform_t)
        spectrogram_t_prime = self.waveform_to_spectrogram(waveform_t_prime)

        # ---- labels from both motion cues ----
        labels_sp    = self.extract_labels_from_json(json_data, motion_key='visual_motion')
        labels_loftr = self.extract_labels_from_json(json_data, motion_key='visual_motion_loftr')

        # ---- IID features (shared) ----
        iid = self.extract_iid_features(json_data)

        return {
            # spectrograms shared by both models
            'spectrogram_t':       spectrogram_t,
            'spectrogram_t_prime': spectrogram_t_prime,

            # labels for SuperPoint/SuperGlue model
            'sp_rotation_direction':    labels_sp['rotation_direction'],
            'sp_translation_direction': labels_sp['translation_direction'],
            'sp_rotation_class':        labels_sp['rotation_class'],
            'sp_translation_class':     labels_sp['translation_class'],

            # labels for LoFTR model
            'loftr_rotation_direction':    labels_loftr['rotation_direction'],
            'loftr_translation_direction': labels_loftr['translation_direction'],
            'loftr_rotation_class':        labels_loftr['rotation_class'],
            'loftr_translation_class':     labels_loftr['translation_class'],

            # shared IID / side labels
            'iid_energy':    iid['temporal_iid_energy'],
            'mean_iid_db':   iid['mean_iid_db'],
            'dominant_side': iid['dominant_side_encoded'],

            'video_id': video_id,
            'clip_id':  clip_id,
        }

    # ------------------------------------------------------------------ helpers

    def waveform_to_spectrogram(self, waveform):
        n_fft = 512; hop_length = 256
        spectrograms = []
        for ch in range(2):
            stft = torch.stft(waveform[ch], n_fft=n_fft, hop_length=hop_length,
                              return_complex=True, window=torch.hann_window(n_fft))
            mag   = torch.log1p(torch.abs(stft))
            phase = torch.angle(stft)
            for arr in (mag, phase):
                arr = torch.nn.functional.interpolate(
                    arr.unsqueeze(0).unsqueeze(0), size=(256, 256),
                    mode='bilinear', align_corners=False).squeeze()
                spectrograms.append(arr)
        spec = torch.stack(spectrograms, dim=0)
        for i in range(spec.shape[0]):
            std = spec[i].std()
            if std > 0:
                spec[i] = (spec[i] - spec[i].mean()) / std
        return spec

    def extract_labels_from_json(self, json_data, motion_key='visual_motion', n_classes=32):
        vm = json_data.get(motion_key, {})
        left, right = vm.get('left', 0), vm.get('right', 0)
        up,   down  = vm.get('up',   0), vm.get('down',  0)

        h = right - left
        v = down  - up

        angle = np.degrees(np.arctan2(v, h)) if (h != 0 or v != 0) else 0.0
        mag   = np.sqrt(h**2 + v**2)

        rot_dir   = int((angle % 360) / (360 / n_classes)) % n_classes
        rot_cls   = min(int(mag / 20), n_classes - 1)
        trans_dir = rot_dir
        trans_cls = rot_cls

        return {
            'rotation_direction':    rot_dir,
            'translation_direction': trans_dir,
            'rotation_class':        rot_cls,
            'translation_class':     trans_cls,
        }

    def extract_iid_features(self, json_data):
        sp = json_data.get('audio_spatial', {})
        side_enc = {'left': 0, 'center': 1, 'right': 2}
        return {
            'temporal_iid_energy':   sp.get('temporal_iid_energy', 0.0),
            'mean_iid_db':           sp.get('mean_iid_db', 0.0),
            'dominant_side_encoded': side_enc.get(sp.get('dominant_side', 'center'), 1),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpatialAudioEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class SpatialSoundLocalizationModel(nn.Module):
    def __init__(self, n_classes=32, in_channels=4):
        super().__init__()
        self.audio_encoder = SpatialAudioEncoder(in_channels=in_channels)
        d = self.audio_encoder.feature_dim

        self.localization_head = nn.Sequential(
            nn.Linear(d + 2, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))

        self.relative_rotation_head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))

        self.relative_translation_head = nn.Sequential(
            nn.Linear(d * 2, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))

        self.rotation_magnitude_head    = nn.Linear(d * 2, n_classes)
        self.translation_magnitude_head = nn.Linear(d * 2, n_classes)

        self.dominant_side_head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, spec_t, spec_t_prime, iid_energy=None, mean_iid_db=None):
        f_t  = self.audio_encoder(spec_t)
        f_tp = self.audio_encoder(spec_t_prime)

        if iid_energy is not None and mean_iid_db is not None:
            iid = torch.stack([iid_energy, mean_iid_db], dim=1)
        else:
            iid = torch.zeros(f_t.size(0), 2, device=f_t.device)

        loc_t  = self.localization_head(torch.cat([f_t,  iid], dim=1))
        loc_tp = self.localization_head(torch.cat([f_tp, iid], dim=1))
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
    ce = nn.CrossEntropyLoss()
    L_rot   = ce(outputs['relative_rotation'],    targets['rotation_direction'])
    L_trans = ce(outputs['relative_translation'], targets['translation_direction'])
    # L_bin   = ce(outputs['rotation_magnitude'],   targets['rotation_class']) + \
    #           ce(outputs['translation_magnitude'], targets['translation_class'])
    L_side  = ce(outputs['dominant_side'],        targets['dominant_side'])

    total = lambda1 * L_rot + (1 - lambda1) * L_trans + lambda2 * L_side
    return total, {
        'rotation':     L_rot.item(),
        'translation':  L_trans.item(),
        # 'binary':       L_bin.item(),
        'dominant_side': L_side.item(),
        'total':        total.item(),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_targets(batch, prefix, device):
    """Pull the correct label set ('sp_' or 'loftr_') from the batch."""
    return {
        'rotation_direction':    batch[f'{prefix}rotation_direction'].to(device),
        'translation_direction': batch[f'{prefix}translation_direction'].to(device),
        'rotation_class':        batch[f'{prefix}rotation_class'].to(device),
        'translation_class':     batch[f'{prefix}translation_class'].to(device),
        'dominant_side':         batch['dominant_side'].to(device),
    }


def _run_epoch(models_dict, loader, optimizers_dict, device,
               lambda1, lambda2, train=True):
    """
    Iterate ONE shared loader.  For each batch, forward/backward through
    BOTH models using their respective label prefixes.
    """
    label_prefix = {'sp': 'sp_', 'loftr': 'loftr_'}

    stats = {
        name: {
            'losses': {k: 0.0 for k in ('rotation', 'translation', 'dominant_side', 'total')},
            'correct': 0,
            'total':   0,
        }
        for name in models_dict
    }

    for name, model in models_dict.items():
        model.train() if train else model.eval()

    for batch in loader:
        spec_t       = batch['spectrogram_t'].to(device)
        spec_t_prime = batch['spectrogram_t_prime'].to(device)
        iid_energy   = batch['iid_energy'].to(device).float()
        mean_iid_db  = batch['mean_iid_db'].to(device).float()

        for name, model in models_dict.items():
            targets = _make_targets(batch, label_prefix[name], device)

            if train:
                optimizers_dict[name].zero_grad()

            if train:
                outputs = model(spec_t, spec_t_prime, iid_energy, mean_iid_db)
                loss, loss_dict = combined_loss(outputs, targets, lambda1, lambda2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizers_dict[name].step()
            else:
                with torch.no_grad():
                    outputs = model(spec_t, spec_t_prime, iid_energy, mean_iid_db)
                    loss, loss_dict = combined_loss(outputs, targets, lambda1, lambda2)

            for k in stats[name]['losses']:
                stats[name]['losses'][k] += loss_dict[k]

            _, pred = torch.max(outputs['relative_rotation'], 1)
            stats[name]['total']   += targets['rotation_direction'].size(0)
            stats[name]['correct'] += (pred == targets['rotation_direction']).sum().item()

    n_batches = len(loader)
    for name in models_dict:
        for k in stats[name]['losses']:
            stats[name]['losses'][k] /= n_batches
        stats[name]['acc'] = 100.0 * stats[name]['correct'] / max(stats[name]['total'], 1)

    return stats


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_dual_models(model_sp, model_loftr,
                      train_loader, val_loader,
                      num_epochs=100, lambda1=0.9, lambda2=1.0,
                      device='cuda', save_dir='.'):
    """
    Train model_sp   (supervised by visual_motion)
    and   model_loftr (supervised by visual_motion_loftr)
    concurrently using a single shared DataLoader each epoch.

    Saved files
    -----------
    best_sp_model.pth       — best val accuracy for SP model
    best_loftr_model.pth    — best val accuracy for LoFTR model
    final_sp_model.pth      — weights at last epoch for SP model
    final_loftr_model.pth   — weights at last epoch for LoFTR model
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models_dict = {'sp': model_sp.to(device), 'loftr': model_loftr.to(device)}
    optimizers_dict = {
        name: optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
        for name, m in models_dict.items()
    }
    schedulers_dict = {
        name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for name, opt in optimizers_dict.items()
    }

    best_val_acc = {'sp': 0.0, 'loftr': 0.0}
    last_val_acc = {'sp': 0.0, 'loftr': 0.0}
    last_train_acc = {'sp': 0.0, 'loftr': 0.0}

    for epoch in range(num_epochs):
        print(f'\n{"="*70}')
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'{"="*70}')
        start_time = time.time()
        train_stats = _run_epoch(models_dict, train_loader, optimizers_dict,
                                 device, lambda1, lambda2, train=True)
        val_stats   = _run_epoch(models_dict, val_loader,   optimizers_dict,
                                 device, lambda1, lambda2, train=False)
        end_time = time.time()
        print(f'Epoch time: {end_time - start_time:.2f} seconds')
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
                    'motion_key':           'visual_motion' if name == 'sp' else 'visual_motion_loftr',
                    'model_state_dict':     models_dict[name].state_dict(),
                    'optimizer_state_dict': optimizers_dict[name].state_dict(),
                    'val_acc':              va,
                    'train_acc':            ta,
                    'lambda1': lambda1, 'lambda2': lambda2
                }, save_dir / f'best_{name}_model.pth')
                print(f'  *** Best {tag} model saved  (val_acc: {va:.2f}%) ***')

        for sched in schedulers_dict.values():
            sched.step()

    # ---- final checkpoints ----
    print(f'\n{"="*70}')
    for name in ('sp', 'loftr'):
        tag  = 'SuperPoint/SuperGlue' if name == 'sp' else 'LoFTR'
        path = save_dir / f'final_{name}_model.pth'
        torch.save({
            'epoch':                num_epochs - 1,
            'motion_cue':           name,
            'motion_key':           'visual_motion' if name == 'sp' else 'visual_motion_loftr',
            'model_state_dict':     models_dict[name].state_dict(),
            'optimizer_state_dict': optimizers_dict[name].state_dict(),
            'val_acc':              last_val_acc[name],
            'train_acc':            last_train_acc[name],
            'lambda1': lambda1, 'lambda2': lambda2
        }, path)
        print(f'[{tag}]  final → {path}')
        print(f'[{tag}]  best  → {save_dir}/best_{name}_model.pth  '
              f'(best val_acc: {best_val_acc[name]:.2f}%)')

    return models_dict['sp'], models_dict['loftr']


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_classes  = 32
    batch_size = 128
    num_epochs = 100
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    lambda1, lambda2 = 0.9, 1.0

    MANIFEST_PATH = Path("/fs/nexus-scratch/vyomwal5/vgg/shortlisted/index.json")
    PROCESSED_DIR = Path("/fs/nexus-scratch/vyomwal5/vgg/processed")
    SAVE_DIR      = Path("/fs/nexus-scratch/vyomwal5/vgg/checkpoints/bs_128")

    # Single dataset — __getitem__ returns labels for BOTH motion cues
    full_dataset = AudioVideoDataset(MANIFEST_PATH, PROCESSED_DIR)
    print(f"Dataset size: {len(full_dataset)} clips")

    val_size   = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator)

    # One shared loader per split — both models see identical batches each step
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              generator=torch.Generator().manual_seed(42))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Two independent models — same architecture, different motion supervision
    model_sp    = SpatialSoundLocalizationModel(n_classes=n_classes, in_channels=4)
    model_loftr = SpatialSoundLocalizationModel(n_classes=n_classes, in_channels=4)

    print(f"Parameters per model: {sum(p.numel() for p in model_sp.parameters()):,}")
    print(f"Device: {device}")
    print(f"  model_sp    ← visual_motion          (SuperPoint/SuperGlue)")
    print(f"  model_loftr ← visual_motion_loftr    (LoFTR)")
    print(f"Checkpoints → {SAVE_DIR}\n")

    trained_sp, trained_loftr = train_dual_models(
        model_sp, model_loftr,
        train_loader, val_loader,
        num_epochs=num_epochs,
        lambda1=lambda1, lambda2=lambda2,
        device=device,
        save_dir=SAVE_DIR,
    )