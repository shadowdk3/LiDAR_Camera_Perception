import torch
from torch.utils.data import DataLoader
from utils import frustum_utils
import numpy as np
import os

def evaluate_full_validation_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    yolo_model_path = "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt"
    frustum_model_path = "checkpoints/frustum_pointnet_checkpoint.pth"

    # Load model weights
    model = frustum_utils.SimpleFrustumPointNet().to(device)
    checkpoint = torch.load(frustum_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create validation dataset and DataLoader (shuffle=False ensures index order matches cached_data)
    dataset = frustum_utils.KittiFrustumDataset(data_path, yolo_model_path)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    total_bev_ious = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_pts, batch_gt_boxes in val_loader:
            batch_pts = batch_pts.to(device)
            
            # Model predicts relative boxes (B, 7)
            pred_boxes_local = model(batch_pts).cpu().numpy()
            gt_boxes_local = batch_gt_boxes.numpy()
            
            # Restore absolute coordinates sample by sample and compute BEV IoU
            for i in range(len(batch_pts)):
                global_dataset_idx = sample_idx + i
                frustum_pts = dataset.cached_data[global_dataset_idx]['frustum_pts']
                centroid = np.mean(frustum_pts, axis=0)
                
                # 1. Add centroid to the predicted relative box center to restore global absolute coordinates
                pred_box_abs = pred_boxes_local[i].copy()
                pred_box_abs[:3] += centroid
                
                # 2. Dataset's __getitem__ already subtracted centroid from gt_box, add it back to restore absolute coordinates
                gt_box_abs = gt_boxes_local[i].copy()
                gt_box_abs[:3] += centroid
                
                # 3. Compute BEV IoU
                # print(f"Sample {global_dataset_idx}:")
                # print(f"  Centroid: {centroid}")
                # print(f"  Pred Box (Abs): {pred_box_abs[:3]}")
                # print(f"  GT Box (Abs):   {gt_box_abs[:3]}")
            
                iou = frustum_utils.compute_bev_iou(pred_box_abs, gt_box_abs)
                total_bev_ious.append(iou)
            
            sample_idx += len(batch_pts)
            
    # Compute mean BEV IoU across the validation set
    if len(total_bev_ious) > 0:
        print("Sample IoU values:", total_bev_ious[:10])
        valid_ious = [iou for iou in total_bev_ious if not np.isnan(iou)]
        mean_bev_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
        print(f"Validation Mean BEV IoU: {mean_bev_iou:.4f}")
    else:
        print("Validation Mean BEV IoU: 0.0000 (No IoU recorded)")
        
if __name__ == "__main__":
    evaluate_full_validation_set()