import torch

# Check MPS availability
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()

print(f"MPS Available: {mps_available}")  # Should be True
print(f"MPS Built: {mps_built}")  # Should be True
print(f"PyTorch Version: {torch.__version__}")
