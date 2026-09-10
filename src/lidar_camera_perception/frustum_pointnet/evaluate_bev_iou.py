import torch
from torch.utils.data import DataLoader
from utils import frustum_utils
import numpy as np
import os

def compute_3d_iou(pred_box, gt_box):
    """
    Computes 3D bounding box IoU (Intersection over Union).
    Boxes format: [x, y, z, l, w, h, yaw] or [x, y, z, dx, dy, dz, yaw]
    """
    try:
        from shapely.geometry import Polygon
        import numpy as np

        def get_poly(box):
            x, y, z, l, w, h, yaw = box[:7]
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            # 4 corners of the 2D footprint
            dx, dy = l / 2.0, w / 2.0
            corners = np.array([
                [dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]
            ])
            R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            rot_corners = np.dot(corners, R.T) + np.array([x, y])
            return Polygon(rot_corners)

        poly_pred = get_poly(pred_box)
        poly_gt = get_poly(gt_box)
        
        if not poly_pred.is_valid or not poly_gt.is_valid:
            return 0.0

        inter_area = poly_pred.intersection(poly_gt).area
        union_area = poly_pred.union(poly_gt).area
        
        if union_area == 0:
            return 0.0
        
        # 3D IoU approximation using height overlap and 2D footprint IoU
        base_iou = inter_area / union_area
        
        pred_z, pred_h = pred_box[2], pred_box[5]
        gt_z, gt_h = gt_box[2], gt_box[5]
        
        pred_z_min, pred_z_max = pred_z - pred_h / 2.0, pred_z + pred_h / 2.0
        gt_z_min, gt_z_max = gt_z - gt_h / 2.0, gt_z + gt_h / 2.0
        
        inter_z_min = max(pred_z_min, gt_z_min)
        inter_z_max = min(pred_z_max, gt_z_max)
        inter_h = max(0.0, inter_z_max - inter_z_min)
        
        pred_vol = poly_pred.area * pred_h
        gt_vol = poly_gt.area * gt_h
        inter_vol = inter_area * inter_h
        union_vol = max(1e-8, pred_vol + gt_vol - inter_vol)
        
        return inter_vol / union_vol
    except Exception:
        return 0.0
    
def evaluate_full_validation_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    yolo_model_path = "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt"
    frustum_model_path = "checkpoints/frustum_pointnet_checkpoint_fine_tune.pth"

    # Load model weights
    model = frustum_utils.FrustumPointNetV2().to(device)
    checkpoint = torch.load(frustum_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create validation dataset and DataLoader (shuffle=False ensures index order matches cached_data)
    dataset = frustum_utils.KittiFrustumDataset(data_path, yolo_model_path)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    total_bev_ious = []
    total_3d_ious = []
    center_errors = []
    heading_errors = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch_pts, batch_gt_boxes, batch_mark in val_loader:
            batch_pts = batch_pts.to(device)
            
            outputs = model(batch_pts)
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            pred_boxes_local = preds.detach().cpu().numpy()
            gt_boxes_local = batch_gt_boxes.numpy()
            
            batch_size = len(batch_pts)
            for i in range(batch_size):
                global_dataset_idx = sample_idx + i
                frustum_pts = dataset.cached_data[global_dataset_idx]['frustum_pts']
                centroid = np.mean(frustum_pts, axis=0)
                
                # Restore absolute coordinates
                pred_box_abs = pred_boxes_local[i].copy()
                pred_box_abs[:3] += centroid
                
                gt_box_abs = gt_boxes_local[i].copy()
                gt_box_abs[:3] += centroid
                
                # Compute BEV IoU
                bev_iou = frustum_utils.compute_bev_iou(pred_box_abs, gt_box_abs)
                total_bev_ious.append(bev_iou)

                # Compute 3D IoU
                iou_3d = compute_3d_iou(pred_box_abs, gt_box_abs)
                total_3d_ious.append(iou_3d)

                # Compute Center Distance Error (MAE for x, y, z)
                center_err = np.linalg.norm(pred_box_abs[:3] - gt_box_abs[:3])
                center_errors.append(center_err)

                # Compute Orientation Error (Yaw angle difference normalized to [-pi, pi])
                if len(pred_box_abs) >= 7 and len(gt_box_abs) >= 7:
                    yaw_diff = np.abs(np.arctan2(np.sin(pred_box_abs[6] - gt_box_abs[6]), 
                                                 np.cos(pred_box_abs[6] - gt_box_abs[6])))
                    heading_errors.append(yaw_diff)
            
            sample_idx += len(batch_pts)
            
    # Compute and display expanded metrics
    if len(total_bev_ious) > 0:
        valid_bev = [i for i in total_bev_ious if not np.isnan(i)]
        valid_3d = [i for i in total_3d_ious if not np.isnan(i)]
        valid_center = [i for i in center_errors if not np.isnan(i)]
        valid_heading = [i for i in heading_errors if not np.isnan(i)]

        print(f"Validation Mean BEV IoU: {sum(valid_bev) / len(valid_bev):.4f}" if valid_bev else "Validation Mean BEV IoU: 0.0000")
        print(f"Validation Mean 3D IoU: {sum(valid_3d) / len(valid_3d):.4f}" if valid_3d else "Validation Mean 3D IoU: 0.0000")
        print(f"Validation Center Distance Error (MAE): {sum(valid_center) / len(valid_center):.4f} m" if valid_center else "Validation Center Distance Error: 0.0000 m")
        print(f"Validation Mean Heading Error: {np.degrees(sum(valid_heading) / len(valid_heading)):.2f}°" if valid_heading else "Validation Mean Heading Error: 0.00°")
    else:
        print("Validation metrics: 0.0000 (No records found)")
        
if __name__ == "__main__":
    evaluate_full_validation_set()