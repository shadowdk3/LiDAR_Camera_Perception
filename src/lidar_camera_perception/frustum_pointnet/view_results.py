import torch
import numpy as np
from utils import frustum_utils

def view_model_results_with_sample(data_path, yolo_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = frustum_utils.KittiFrustumDataset(data_path, yolo_path)
    model = frustum_utils.FrustumPointNetV2().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"=> Starting result visualization. Total samples: {len(dataset)}")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sampled_pts, gt_box_tensor, batch_mark = dataset[idx]
            raw_sample = dataset.cached_data[idx]
            
            # Re-compute the exact same centroid used during dataset normalization
            frustum_pts = raw_sample['frustum_pts']
            centroid = np.mean(frustum_pts, axis=0)
        
            # Run model inference
            batch_points = sampled_pts.unsqueeze(0).to(device)
            predictions, logits, center_offset = model(batch_points)
            
            pred_box = predictions[0].cpu().numpy()
            
            # Map the prediction back to absolute camera coordinates by adding the centroid
            pred_box[0] += centroid[0]
            pred_box[1] += centroid[1]
            pred_box[2] += centroid[2]
            
            print(f"\nViewing sample {idx + 1} / {len(dataset)}...")
            
            # Call your visualization function with both GT and Prediction
            frustum_utils.visualize_sample(
                bin_path=raw_sample['bin_path'],
                img_path=raw_sample['img_path'],
                gt_box=raw_sample['gt_box'],
                pred_box=pred_box
            )

if __name__ == "__main__":
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    yolo_path = "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt"
    model_path = "checkpoints/frustum_pointnet_checkpoint_fine_tune.pth"
    
    view_model_results_with_sample(data_path, yolo_path, model_path)