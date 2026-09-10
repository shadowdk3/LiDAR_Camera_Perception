import torch
from pathlib import Path
import frustum_utils
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "frustum_pointnet_checkpoint_fine_tune.pth")
output_onnx_path = os.path.join(PROJECT_ROOT, "checkpoints", "frustum_pointnet_fine_tune.onnx")

if not os.path.exists(checkpoint_path):
    print(f"{checkpoint_path} not found!")
    
device = torch.device("cuda")
model = frustum_utils.FrustumPointNetV2().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate a random dummy tensor to represent a single frustum point cloud
# Shape: (Batch_Size=1, Num_Points=512, Channels=3 -> x, y, z)
dummy_input = torch.randn(1, 512, 3, device=device)

torch.onnx.export(
    model, 
    dummy_input, 
    output_onnx_path,
    input_names=['points'],
    output_names=['bounding_box'],
    dynamic_axes={'points': {0: 'batch_size'}, 'bounding_box': {0: 'batch_size'}},
    opset_version=13
)

print("=> Model successfully exported to ONNX.")