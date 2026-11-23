import torch

# Load the checkpoint
checkpoint = torch.load("/fs/nexus-scratch/vyomwal5/vgg/checkpoints/best_sp_model.pth",map_location=torch.device('cpu'))  # Use map_location if loading on a different device

# Extract saved information
epoch = checkpoint['epoch']
motion_cue = checkpoint['motion_cue']
motion_key = checkpoint['motion_key']
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
val_acc = checkpoint['val_acc']
train_acc = checkpoint['train_acc']
lambda1 = checkpoint['lambda1']
lambda2 = checkpoint['lambda2']

# Load state dictionaries into your model and optimizer
# model.load_state_dict(model_state_dict)
# optimizer.load_state_dict(optimizer_state_dict)

print(f"Loaded checkpoint from epoch {epoch}")
print(f"Motion cue: {motion_cue}, Val Acc: {val_acc:.2f}%")
print(f"Motion key: {motion_key}")
# print(f"Model state dict keys: {model_state_dict.keys()}")
# print(f"Optimizer state dict keys: {optimizer_state_dict.keys()}")
print(f"Train Acc: {train_acc:.2f}%")
print(f"Lambda1: {lambda1}, Lambda2: {lambda2}")