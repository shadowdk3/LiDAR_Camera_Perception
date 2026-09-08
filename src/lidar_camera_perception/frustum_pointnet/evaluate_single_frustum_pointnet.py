import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from utils import frustum_utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    img_path = os.path.join(data_dir, "image_00/data/0000000400.png")
    bin_path = os.path.join(data_dir, "velodyne_points/data/0000000400.bin")
    label_path = os.path.join(data_dir, "tracklet_labels.xml")
    log_path = 'runs/frustum_pointnet_eval'
    model_path = "checkpoints/frustum_pointnet_checkpoint.pth"
    
    img = cv2.imread(img_path)
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    annotations = frustum_utils.parse_kitti_tracklets(label_path)
    proj, ext, cam0_to_cam2_proj = frustum_utils.read_kitti_calib()
    
    num_epochs = 100  # Define your total epoch count here
    
    # Load trained model weights
    model = frustum_utils.SimpleFrustumPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    corner_loss_fn = frustum_utils.Custom3DCornerLoss().to(device)
        
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.eval()
    
    # Initialize TensorBoard writer for evaluation/inference logging
    writer = SummaryWriter(log_dir=log_path)
    
    yolo = YOLO("/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    results = yolo(img, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    confss = results.boxes.conf.cpu().numpy() # Obtain the confidence scores of the YOLO 2D bounding boxes
    
    current_frame_idx = int(os.path.basename(img_path).split('.')[0])
    
    total_eval_loss = 0.0
    valid_detections_count = 0
    
    pred_boxes_list = []
    scores_list = []
    
    # 1. Run inference and draw Predicted 3D Boxes (Red)
    with torch.no_grad():
        for box2d, cls_id, conf in zip(boxes, clss, confss):
            if int(cls_id) in frustum_utils.TARGET_CLASSES.values():
                pts_3d = point_cloud[:, :3]
                pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
                pts_cam = (ext @ pts_homo.T).T[:, :3]
                valid_mask = pts_cam[:, 2] > 0.1
                
                pts_2d_homo = (proj @ pts_homo[valid_mask].T).T
                u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
                v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
                
                u1, v1, u2, v2 = box2d
                in_box = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
                frustum_pts = pts_3d[valid_mask][in_box]
                
                if len(frustum_pts) > 30:
                    centroid = np.mean(frustum_pts, axis=0)
                    norm_pts = frustum_pts - centroid
                    
                    if len(norm_pts) >= 512:
                        choice = np.random.choice(len(norm_pts), 512, replace=False)
                    else:
                        choice = np.random.choice(len(norm_pts), 512, replace=True)
                    sampled_pts = norm_pts[choice]
                    
                    tensor_input = torch.tensor(sampled_pts, dtype=torch.float32).unsqueeze(0).to(device) # (1, 512, 3)
                    pred_box = model(tensor_input).squeeze(0).cpu().numpy() # (7,)
                    
                    # Transform predicted relative box back to global/camera coordinate frame by adding centroid
                    pred_box_global = pred_box.copy()
                    pred_box[:3] += centroid
                    
                    pred_boxes_list.append(pred_box)
                    scores_list.append(conf) # Use YOLO's confidence scores as the sorting criteria
                    
        # 2. Perform 3D NMS to filter out overlapping bounding boxes
        print(f"Collected boxes count: {len(pred_boxes_list)}")
        if len(pred_boxes_list) > 0:
            pred_boxes_arr = np.array(pred_boxes_list)
            scores_arr = np.array(scores_list)
            
            keep_indices = frustum_utils.nms_3d(pred_boxes_arr, scores_arr, iou_threshold=0.15)
            final_boxes = pred_boxes_arr[keep_indices]            
                    
            # 3. Draw filtered clean prediction boxes (Red)
            for box in final_boxes:
                img = frustum_utils.draw_3d_box(img, box, proj, (0, 0, 255))        
                img_h, img_w = img.shape[:2]
                
                best_loss = float('inf')
                # Compute evaluation loss against matched ground truth if available
                for tracklet in annotations:
                    pose_idx = current_frame_idx - tracklet['first_frame']
                    if 0 <= pose_idx < len(tracklet['poses']):
                        p = tracklet['poses'][pose_idx]
                        
                        # Check if the GT center point is within the 2D image view
                        gt_center_homo = np.array([p['tx'], p['ty'], p['tz'], 1.0])
                        proj_pt = proj @ gt_center_homo
                        if proj_pt[2] <= 0:
                            continue
                        u = proj_pt[0] / proj_pt[2]
                        v = proj_pt[1] / proj_pt[2]
                        if not (0 <= u < img_w and 0 <= v < img_h):
                            continue # Skip GTs outside the image view
                        
                        gt_box = np.array([p['tx'], p['ty'], p['tz'], tracklet['l'], tracklet['w'], tracklet['h'], p['rz']], dtype=np.float32)
                        
                        pred_tensor = torch.tensor(pred_box, dtype=torch.float32).unsqueeze(0)
                        gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
                        
                        # 1. Center loss (x, y, z)
                        loss_center = nn.SmoothL1Loss()(pred_tensor[:, :3], gt_tensor[:, :3]).item()
            
                        # 2. Size loss (l, w, h) - Add this to prevent shrinking
                        loss_size = nn.SmoothL1Loss()(pred_tensor[:, 3:6], gt_tensor[:, 3:6]).item()

                        # 3. Corner loss for overall geometry and rotation
                        loss_corner = corner_loss_fn(pred_tensor, gt_tensor).item()
                        
                        # Combined Loss 
                        loss = loss_center + loss_size + loss_corner

                        if loss < best_loss:
                            best_loss = loss
                        
                if best_loss != float('inf'):
                    total_eval_loss += best_loss
                    valid_detections_count += 1
                 
        # 4. Draw Ground Truth 3D Boxes (Green) for comparison
        for tracklet in annotations:
            pose_idx = current_frame_idx - tracklet['first_frame']
            if 0 <= pose_idx < len(tracklet['poses']):
                p = tracklet['poses'][pose_idx]
                
                # Check similarly whether it is within the image view to avoid drawing weird projections
                gt_center_homo = np.array([p['tx'], p['ty'], p['tz'], 1.0])
                proj_pt = proj @ gt_center_homo
                if proj_pt[2] <= 0:
                    continue
                
                u = proj_pt[0] / proj_pt[2]
                v = proj_pt[1] / proj_pt[2]
                if 0 <= u < img_w and 0 <= v < img_h:
                    gt_box = np.array([p['tx'], p['ty'], p['tz'], tracklet['l'], tracklet['w'], tracklet['h'], p['rz']], dtype=np.float32)
                    # Draw GT in Green -> (0, 255, 0)
                    img = frustum_utils.draw_3d_box(img, gt_box, proj, (0, 255, 0))
                    
    # Plot and compare 3D point cloud with BEV (Bird's Eye View)
    plt.figure(figsize=(8, 8))
    # Plot current frame's LiDAR point cloud (x, y plane)
    pts_3d = point_cloud[:, :3]
    plt.scatter(pts_3d[:, 0], pts_3d[:, 1], s=0.5, c='gray', label='LiDAR Points')

    # Plot predicted 3D box centers (Red)
    for box in final_boxes:
        plt.scatter(box[0], box[1], c='red', marker='x', s=120, linewidths=3, label='Pred Center' if 'Pred Center' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot ground truth (GT) 3D box centers (Green)
    for tracklet in annotations:
        pose_idx = current_frame_idx - tracklet['first_frame']
        if 0 <= pose_idx < len(tracklet['poses']):
            p = tracklet['poses'][pose_idx]
            gt_center_homo = np.array([p['tx'], p['ty'], p['tz'], 1.0])
            proj_pt = proj @ gt_center_homo
            if proj_pt[2] > 0:
                plt.scatter(p['tx'], p['ty'], c='green', marker='o', s=100, label='GT Center' if 'GT Center' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlim(0, 45)    # 0 to 45 meters forward from the vehicle
    plt.ylim(-20, 20)  # 20 meters left and right each
    plt.xlabel('X (Forward meters)')
    plt.ylabel('Y (Left meters)')
    plt.title('BEV (Bird\'s-Eye-View) Space: Red(Pred) vs Green(GT)')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show() 
        
    if valid_detections_count > 0:
        avg_eval_loss = total_eval_loss / valid_detections_count
        print(f"=> Evaluation Smooth L1 Loss: {avg_eval_loss:.4f}")
        writer.add_scalar('Loss/Eval', avg_eval_loss, 0)
        
    # Log visualized image frame directly into TensorBoard
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    writer.add_image('Visual/Predictions_vs_GT', img_rgb, 0, dataformats='HWC')
    
    writer.close()
    
    cv2.imshow("Frustum PointNet 3D Detection (Red: Pred | Green: GT)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(
        1
    )  # Forces OpenCV to flush the GUI event queue and release terminal lock