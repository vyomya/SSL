import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from torchvision import models
import json
from pathlib import Path


class AudioVideoDataset(Dataset):
    """Dataset for temporal audio pairs with pre-calculated camera motion and IID audio features"""
    def __init__(self, audio_paths, processed_base_dir, sample_rate=48000, clip_duration=2.5):
        """
        Args:
            audio_paths: List of audio file paths
            processed_base_dir: Base directory containing processed JSON files
            sample_rate: Audio sample rate (default 48000)
            clip_duration: Duration of each audio clip (2.5s for time t and t')
        """
        self.audio_paths = audio_paths
        self.processed_base_dir = Path(processed_base_dir)
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_length = int(sample_rate * clip_duration)
        
    def __len__(self):
        return len(self.audio_paths)
    
    def parse_filename(self, filepath):
        """
        Parse filename like '--0PQM4-hqg_30_40_clip_0000_0005.wav'
        Returns: video_id ('--0PQM4-hqg_30_40'), clip_id ('clip_0000_0005')
        """
        filename = Path(filepath).stem  # Remove extension
        # Split by '_clip_' to separate video_id and clip_id
        parts = filename.split('_clip_')
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")
        
        video_id = parts[0]
        clip_id = f"clip_{parts[1]}"
        
        return video_id, clip_id
    
    def load_json_data(self, video_id, clip_id):
        """
        Load pre-calculated motion and audio data from JSON
        Path: processed_base_dir/video_id/clip_id/result.json
        """
        json_path = self.processed_base_dir / video_id / clip_id / "result.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __getitem__(self, idx):
        # Parse filename to get video_id and clip_id
        video_id, clip_id = self.parse_filename(self.audio_paths[idx])
        
        # Load pre-calculated data
        json_data = self.load_json_data(video_id, clip_id)
        
        # Load full 5-second audio
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Handle mono by duplicating to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Split into two temporal clips: time t and time t'
        # time t: [0, 2.5s], time t': [2.5s, 5s]
        total_length = waveform.shape[1]
        mid_point = total_length // 2
        
        waveform_t = waveform[:, :mid_point]
        waveform_t_prime = waveform[:, mid_point:mid_point*2]
        
        # Ensure proper length for both clips
        if waveform_t.shape[1] < self.clip_length:
            waveform_t = torch.nn.functional.pad(waveform_t, (0, self.clip_length - waveform_t.shape[1]))
        else:
            waveform_t = waveform_t[:, :self.clip_length]
            
        if waveform_t_prime.shape[1] < self.clip_length:
            waveform_t_prime = torch.nn.functional.pad(waveform_t_prime, (0, self.clip_length - waveform_t_prime.shape[1]))
        else:
            waveform_t_prime = waveform_t_prime[:, :self.clip_length]
        
        # Convert both to spectrograms
        spectrogram_t = self.waveform_to_spectrogram(waveform_t)
        spectrogram_t_prime = self.waveform_to_spectrogram(waveform_t_prime)
        
        # Extract labels from JSON data
        labels = self.extract_labels_from_json(json_data)
        
        # Get IID audio features
        iid_features = self.extract_iid_features(json_data)
        
        return {
            'spectrogram_t': spectrogram_t,
            'spectrogram_t_prime': spectrogram_t_prime,
            'rotation_direction': labels['rotation_direction'],
            'translation_direction': labels['translation_direction'],
            'rotation_class': labels['rotation_class'],
            'translation_class': labels['translation_class'],
            'iid_energy': iid_features['temporal_iid_energy'],
            'mean_iid_db': iid_features['mean_iid_db'],
            'dominant_side': iid_features['dominant_side_encoded'],
            'video_id': video_id,
            'clip_id': clip_id
        }
    
    def waveform_to_spectrogram(self, waveform):
        """Convert 2-channel waveform to 256x256x4 spectrogram"""
        n_fft = 512
        hop_length = 256
        
        spectrograms = []
        for channel in range(2):
            stft = torch.stft(
                waveform[channel],
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(n_fft)
            )
            
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Log scale
            magnitude = torch.log1p(magnitude)
            
            # Resize to 256x256
            magnitude = torch.nn.functional.interpolate(
                magnitude.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            phase = torch.nn.functional.interpolate(
                phase.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            spectrograms.append(magnitude)
            spectrograms.append(phase)
        
        spectrogram = torch.stack(spectrograms, dim=0)
        
        # Normalize
        for i in range(spectrogram.shape[0]):
            mean = spectrogram[i].mean()
            std = spectrogram[i].std()
            if std > 0:
                spectrogram[i] = (spectrogram[i] - mean) / std
        
        return spectrogram
    
    def extract_labels_from_json(self, json_data, n_classes=32):
        """
        Extract rotation and translation labels from pre-calculated visual motion
        JSON format:
        {
            "visual_motion": {
                "left": 380,
                "right": 96,
                "up": 346,
                "down": 130
            }
        }
        """
        visual_motion = json_data.get('visual_motion', {})
        
        left = visual_motion.get('left', 0)
        right = visual_motion.get('right', 0)
        up = visual_motion.get('up', 0)
        down = visual_motion.get('down', 0)
        
        # Calculate horizontal motion (left-right)
        horizontal_motion = right - left
        
        # Calculate vertical motion (up-down)
        vertical_motion = down - up
        
        # Rotation direction class (based on horizontal motion)
        # Map to 0-360 degrees range
        if horizontal_motion != 0 or vertical_motion != 0:
            rotation_angle = np.degrees(np.arctan2(vertical_motion, horizontal_motion))
        else:
            rotation_angle = 0
        
        rotation_direction = int((rotation_angle % 360) / (360 / n_classes)) % n_classes
        
        # Rotation magnitude class (total motion magnitude)
        rotation_magnitude = np.sqrt(horizontal_motion**2 + vertical_motion**2)
        # Normalize and discretize
        rotation_class = min(int(rotation_magnitude / 20), n_classes - 1)  # Adjust scaling as needed
        
        # Translation direction (primarily horizontal)
        translation_angle = rotation_angle  # Same as rotation for this simplified version
        translation_direction = int((translation_angle % 360) / (360 / n_classes)) % n_classes
        
        # Translation magnitude class
        translation_magnitude = rotation_magnitude  # Same as total motion
        translation_class = min(int(translation_magnitude / 20), n_classes - 1)
        
        return {
            'rotation_direction': rotation_direction,
            'translation_direction': translation_direction,
            'rotation_class': rotation_class,
            'translation_class': translation_class
        }
    
    def extract_iid_features(self, json_data):
        """
        Extract IID (Interaural Intensity Difference) features from JSON
        JSON format:
        {
            "audio_spatial": {
                "temporal_iid_energy": 0.475,
                "mean_iid_db": 0.284,
                "dominant_side": "right",
                "iid_per_second": [...],
                "num_seconds": 5
            }
        }
        """
        audio_spatial = json_data.get('audio_spatial', {})
        
        temporal_iid_energy = audio_spatial.get('temporal_iid_energy', 0.0)
        mean_iid_db = audio_spatial.get('mean_iid_db', 0.0)
        dominant_side = audio_spatial.get('dominant_side', 'center')
        
        # Encode dominant side: left=0, center=1, right=2
        side_encoding = {'left': 0, 'center': 1, 'right': 2}
        dominant_side_encoded = side_encoding.get(dominant_side, 1)
        
        return {
            'temporal_iid_energy': temporal_iid_energy,
            'mean_iid_db': mean_iid_db,
            'dominant_side_encoded': dominant_side_encoded
        }


class SpatialAudioEncoder(nn.Module):
    """Spatial Audio Encoder with ResNet-18 backbone (shared weights)"""
    def __init__(self, in_channels=4):
        super().__init__()
        
        # ResNet-18 backbone
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify first conv for 4-channel input (stereo magnitude + phase)
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove final FC layer to get features
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.feature_dim = num_features
        
    def forward(self, x):
        """
        Args:
            x: Spectrogram of shape (batch, 4, 256, 256)
        Returns:
            features: Shape (batch, 512)
        """
        return self.resnet(x)


class SpatialSoundLocalizationModel(nn.Module):
    """
    Complete architecture with IID features:
    - Spatial Audio Encoder with shared weights for time t and t'
    - Egomotion from Visual Cues (pre-calculated)
    - IID audio features for spatial localization
    - Relative motion supervision
    """
    def __init__(self, n_classes=32, in_channels=4):
        super().__init__()
        
        # Shared Spatial Audio Encoder
        self.audio_encoder = SpatialAudioEncoder(in_channels=in_channels)
        feature_dim = self.audio_encoder.feature_dim
        
        # Spatial Localization Prediction Head (with IID features)
        # Incorporates IID energy and mean IID as additional features
        self.localization_head = nn.Sequential(
            nn.Linear(feature_dim + 2, 512),  # +2 for IID features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
        # Sound Relative Motion Prediction Heads
        # These predict the relative motion between time t and t'
        self.relative_rotation_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Concatenated features from t and t'
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
        self.relative_translation_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes)
        )
        
        # Rotation and Translation magnitude classification
        self.rotation_magnitude_head = nn.Linear(feature_dim * 2, n_classes)
        self.translation_magnitude_head = nn.Linear(feature_dim * 2, n_classes)
        
        # Dominant side prediction (left/center/right)
        self.dominant_side_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes: left, center, right
        )
        
    def forward(self, spec_t, spec_t_prime, iid_energy=None, mean_iid_db=None):
        """
        Args:
            spec_t: Spectrogram at time t, shape (batch, 4, 256, 256)
            spec_t_prime: Spectrogram at time t', shape (batch, 4, 256, 256)
            iid_energy: IID energy values, shape (batch,)
            mean_iid_db: Mean IID in dB, shape (batch,)
        Returns:
            Dictionary with predictions
        """
        # Extract features using shared encoder
        features_t = self.audio_encoder(spec_t)
        features_t_prime = self.audio_encoder(spec_t_prime)
        
        # Spatial localization predictions with IID features
        if iid_energy is not None and mean_iid_db is not None:
            # Combine audio features with IID features
            iid_features = torch.stack([iid_energy, mean_iid_db], dim=1)  # (batch, 2)
            localization_input_t = torch.cat([features_t, iid_features], dim=1)
            localization_input_t_prime = torch.cat([features_t_prime, iid_features], dim=1)
        else:
            # Fallback without IID features (pad with zeros)
            zeros = torch.zeros(features_t.size(0), 2, device=features_t.device)
            localization_input_t = torch.cat([features_t, zeros], dim=1)
            localization_input_t_prime = torch.cat([features_t_prime, zeros], dim=1)
        
        localization_t = self.localization_head(localization_input_t)
        localization_t_prime = self.localization_head(localization_input_t_prime)
        
        # Concatenate features for relative motion prediction
        combined_features = torch.cat([features_t, features_t_prime], dim=1)
        
        # Relative motion predictions (from t to t')
        relative_rotation = self.relative_rotation_head(combined_features)
        relative_translation = self.relative_translation_head(combined_features)
        
        # Magnitude predictions
        rotation_magnitude = self.rotation_magnitude_head(combined_features)
        translation_magnitude = self.translation_magnitude_head(combined_features)
        
        # Dominant side prediction (from time t features)
        dominant_side_pred = self.dominant_side_head(features_t)
        
        return {
            'localization_t': localization_t,
            'localization_t_prime': localization_t_prime,
            'relative_rotation': relative_rotation,
            'relative_translation': relative_translation,
            'rotation_magnitude': rotation_magnitude,
            'translation_magnitude': translation_magnitude,
            'dominant_side': dominant_side_pred
        }


def combined_loss(outputs, targets, lambda1=0.9, lambda2=1.0, lambda3=0.5):
    """
    Combined loss with IID-based dominant side prediction:
    L = λ1*L_rot + (1-λ1)*L_trans + λ2*L_bin + λ3*L_side
    
    Where:
    - L_rot: Relative rotation direction loss
    - L_trans: Relative translation direction loss  
    - L_bin: Combined magnitude classification loss
    - L_side: Dominant side classification loss (from IID)
    """
    criterion = nn.CrossEntropyLoss()
    
    # Relative motion losses (main supervision from camera motion)
    L_rot = criterion(outputs['relative_rotation'], targets['rotation_direction'])
    L_trans = criterion(outputs['relative_translation'], targets['translation_direction'])
    
    # Magnitude losses
    L_rot_mag = criterion(outputs['rotation_magnitude'], targets['rotation_class'])
    L_trans_mag = criterion(outputs['translation_magnitude'], targets['translation_class'])
    L_bin = L_rot_mag + L_trans_mag
    
    # Dominant side loss (from IID audio features)
    L_side = criterion(outputs['dominant_side'], targets['dominant_side'])
    
    # Combined loss
    total_loss = lambda1 * L_rot + (1 - lambda1) * L_trans + lambda2 * L_bin + lambda3 * L_side
    
    return total_loss, {
        'rotation': L_rot.item(),
        'translation': L_trans.item(),
        'binary': L_bin.item(),
        'dominant_side': L_side.item(),
        'total': total_loss.item()
    }


def train_with_temporal_pairs(model, train_loader, val_loader, 
                              num_epochs=100, lambda1=0.9, lambda2=1.0, lambda3=0.5,
                              device='cuda'):
    """Training loop with temporal audio pairs, camera motion supervision, and IID features"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = {'rotation': 0, 'translation': 0, 'binary': 0, 'dominant_side': 0, 'total': 0}
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            spec_t = batch['spectrogram_t'].to(device)
            spec_t_prime = batch['spectrogram_t_prime'].to(device)
            iid_energy = batch['iid_energy'].to(device).float()
            mean_iid_db = batch['mean_iid_db'].to(device).float()
            
            targets = {
                'rotation_direction': batch['rotation_direction'].to(device),
                'translation_direction': batch['translation_direction'].to(device),
                'rotation_class': batch['rotation_class'].to(device),
                'translation_class': batch['translation_class'].to(device),
                'dominant_side': batch['dominant_side'].to(device)
            }
            
            optimizer.zero_grad()
            
            # Forward pass with temporal pairs and IID features
            outputs = model(spec_t, spec_t_prime, iid_energy, mean_iid_db)
            
            # Compute loss
            loss, loss_dict = combined_loss(outputs, targets, lambda1, lambda2, lambda3)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            for key in train_losses:
                train_losses[key] += loss_dict[key]
            
            # Calculate accuracy for relative rotation prediction
            _, predicted = torch.max(outputs['relative_rotation'], 1)
            train_total += targets['rotation_direction'].size(0)
            train_correct += (predicted == targets['rotation_direction']).sum().item()
        
        train_acc = 100 * train_correct / train_total
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_losses = {'rotation': 0, 'translation': 0, 'binary': 0, 'dominant_side': 0, 'total': 0}
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                spec_t = batch['spectrogram_t'].to(device)
                spec_t_prime = batch['spectrogram_t_prime'].to(device)
                iid_energy = batch['iid_energy'].to(device).float()
                mean_iid_db = batch['mean_iid_db'].to(device).float()
                
                targets = {
                    'rotation_direction': batch['rotation_direction'].to(device),
                    'translation_direction': batch['translation_direction'].to(device),
                    'rotation_class': batch['rotation_class'].to(device),
                    'translation_class': batch['translation_class'].to(device),
                    'dominant_side': batch['dominant_side'].to(device)
                }
                
                outputs = model(spec_t, spec_t_prime, iid_energy, mean_iid_db)
                loss, loss_dict = combined_loss(outputs, targets, lambda1, lambda2, lambda3)
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key]
                
                _, predicted = torch.max(outputs['relative_rotation'], 1)
                val_total += targets['rotation_direction'].size(0)
                val_correct += (predicted == targets['rotation_direction']).sum().item()
        
        val_acc = 100 * val_correct / val_total
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Acc: {train_acc:.2f}% | Total: {train_losses["total"]:.4f} | '
              f'Rot: {train_losses["rotation"]:.4f} | Trans: {train_losses["translation"]:.4f} | '
              f'Bin: {train_losses["binary"]:.4f} | Side: {train_losses["dominant_side"]:.4f}')
        print(f'Val   - Acc: {val_acc:.2f}% | Total: {val_losses["total"]:.4f} | '
              f'Rot: {val_losses["rotation"]:.4f} | Trans: {val_losses["translation"]:.4f} | '
              f'Bin: {val_losses["binary"]:.4f} | Side: {val_losses["dominant_side"]:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'lambda1': lambda1,
                'lambda2': lambda2,
                'lambda3': lambda3
            }, 'best_spatial_localization_model.pth')
            print(f'*** New best model saved! Accuracy: {val_acc:.2f}% ***')
        
        scheduler.step()
    
    return model


# Example usage
if __name__ == "__main__":
    # Configuration
    n_classes = 32
    batch_size = 16  # Reduced from 300 for stability
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyperparameters
    lambda1 = 0.9  # Rotation weight
    lambda2 = 1.0  # Binary (magnitude) weight
    lambda3 = 0.5  # Dominant side weight
    
    # Create model matching the architecture diagram
    model = SpatialSoundLocalizationModel(n_classes=n_classes, in_channels=4)
    
    print("Spatial Sound Localization Model (with pre-calculated motion and IID features)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nKey features:")
    print("✓ Shared Spatial Audio Encoder for time t and t'")
    print("✓ Temporal audio pairs (2.5s each)")
    print("✓ Relative rotation and translation prediction")
    print("✓ Pre-calculated camera motion from Superglue")
    print("✓ IID-based audio spatial features")
    print("✓ Dominant side prediction")
    
    # Paths
    AUDIO_DIR = Path("/Users/vyomagarwal/Documents/projects/sound_localization/VGGSound/audioset_data/shortlisted/audio")
    PROCESSED_DIR = Path("/Users/vyomagarwal/Documents/projects/sound_localization/VGGSound/audioset_data/processed")
    
    audio_paths = sorted([str(v) for v in AUDIO_DIR.iterdir() if v.suffix == '.wav'])
    
    print(f"\nFound {len(audio_paths)} audio files")
    
    # Example dataset
    dataset = AudioVideoDataset(audio_paths, PROCESSED_DIR)
    
    # Train/validation split
    val_fraction = 0.2
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_fraction)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
   
    # Training
    print(f"\nStarting training on {device}...")
    trained_model = train_with_temporal_pairs(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        device=device
    )