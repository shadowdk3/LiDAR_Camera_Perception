"""
Define Dataset (KITTI path parsing, YOLO frustum extraction, padding to 512 pts)
Create DataLoader (batch_size=4, shuffle=True)
Initialize model & optimizer (SimpleFrustumPointNet, Adam lr=0.001)
Training loop: forward pass, loss computation, zero_grad, backward, and step
Save best model weights (lowest loss) to 'frustum_pointnet.pth'
"""

import numpy as np
import cv2
import open3d as o3d
import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

TARGET_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'traffic_light': 9,
    'stop_sign': 11
}

class SimpleFrustumPointNet(nn.Module):
    def __init__(self):
        super(SimpleFrustumPointNet, self).__init__()
        # Shared MLP(Multilayer Perceptron) (PointNet Core Feature Extraction)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Regression Head: Predicts [x, y, z, l, w, h, yaw]
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, points):
        # points shape: (B, N, 3) -> Permute to (B, 3, N) for Conv1d
        x = points.permute(0, 2, 1)
        x = self.mlp1(x) # (B, 1024, N)
        
        # Symmetric Function (Max Pooling) for Global Features
        x = torch.max(x, dim=2)[0] # (B, 1024)
        
        # 3D Bounding Box Parameters Regression
        box_params = self.fc(x) # (B, 7)
        return box_params
    
def read_kitti_calib():
    # 1. Extrinsic Matrix from LiDAR to Cam0 (4x4)
    velo_to_cam0_extrinsic = np.array([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])

    # 2. Rectification Matrix for Cam0 (4x4)
    cam0_rectification = np.array([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])

    # 3. Projection Matrix for Left Color Camera (Cam2) (3x4)
    cam2_projection_rectified = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])

    velo_to_cam2_projection = cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_extrinsic
    return velo_to_cam2_projection, velo_to_cam0_extrinsic, cam0_rectification, cam2_projection_rectified

def parse_kitti_tracklets(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets = []
    
    for item in root.findall('.//item'):
        obj_type_node = item.find('objectType')
        obj_type = obj_type_node.text if obj_type_node is not None else 'Unknown'
        
        h_node = item.find('h')
        w_node = item.find('w')
        l_node = item.find('l')
        ff_node = item.find('first_frame')
        
        tracklet = {
            'objectType': obj_type,
            'h': float(h_node.text) if h_node is not None else 0.0,
            'w': float(w_node.text) if w_node is not None else 0.0,
            'l': float(l_node.text) if l_node is not None else 0.0,
            'first_frame': int(ff_node.text) if ff_node is not None else 0,
            'poses': []
        }
        
        poses_node = item.find('poses')
        if poses_node is not None:
            for pose_item in poses_node.findall('item'):
                tx_node = pose_item.find('tx')
                ty_node = pose_item.find('ty')
                tz_node = pose_item.find('tz')
                rz_node = pose_item.find('rz')
                
                tracklet['poses'].append({
                    'tx': float(tx_node.text) if tx_node is not None else 0.0,
                    'ty': float(ty_node.text) if ty_node is not None else 0.0,
                    'tz': float(tz_node.text) if tz_node is not None else 0.0,
                    'rz': float(rz_node.text) if rz_node is not None else 0.0
                })
        tracklets.append(tracklet)
    return tracklets

def visualize_sample(bin_path, img_path, gt_box, pred_box=None):
    # 1. Load the image and raw LiDAR point cloud data
    img = cv2.imread(img_path)
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts_3d = point_cloud[:, :3]
    
    # 2. Create and populate an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d)
    
    # 3. Helper function to create an Open3D OrientedBoundingBox (OBB) from 7D box parameters
    # box_params format: [x, y, z, l, w, h, rz]
    def create_o3d_box(box_params, color):
        x, y, z, l, w, h, rz = box_params
        print("verify:", [x, y, z, l, w, h, rz])
        # Create a rotation matrix around the Z-axis using the yaw angle (rz)
        rot = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, rz))
        obb = o3d.geometry.OrientedBoundingBox(np.array([x, y, z]), rot, np.array([l, w, h]))
        obb.color = color
        return obb
        
    geometries = [pcd]
    gt_obb = create_o3d_box(gt_box, color=[0, 1, 0]) # Green represents Ground Truth
    geometries.append(gt_obb)
    
    if pred_box is not None:
        pred_obb = create_o3d_box(pred_box, color=[1, 0, 0]) # Red represents Prediction
        geometries.append(pred_obb)
        
    # 4. Open an interactive 3D visualization window (supports mouse rotation, translation, and zoom)
    print("=> Displaying 3D point cloud and bounding boxes (use mouse to rotate, zoom, and pan)")
    o3d.visualization.draw_geometries(geometries)
    
"""
RAM Caching & Data Pipeline Optimization

Pre-Generation (__init__): Runs YOLO once per frame beforehand to index every bounding box proposal separately, 
    avoiding CUDA multi-threading crashes and capturing multiple objects per image.
Frustum Extraction (__getitem__): Keeps your exact point-cloud projection, 2D box filtering, zero-center normalization, 
    and 512-point uniform sampling logic intact.
DataLoader Usage: When instantiating your DataLoader, explicitly set `num_workers=0` to ensure safe interaction
    with YOLO model weights.
"""
class KittiFrustumDataset(Dataset):
    def __init__(self, data_dir, yolo_model_path):
        self.img_dir = os.path.join(data_dir, "image_00/data")
        self.bin_dir = os.path.join(data_dir, "velodyne_points/data")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.annotations = parse_kitti_tracklets(os.path.join(data_dir, "tracklet_labels_cleaned.xml"))
        self.proj, self.ext, cam0_rectification, cam2_projection_rectified = read_kitti_calib()

        # Construct a composite matrix to project from unrectified Camera 0 coordinates to Camera 2 (RGB image) pixels.
        # 1. cam0_rectification (R0_rect): Rectifies the raw Camera 0 coordinate system 
        #    to align with the standard rectified coordinate system used by KITTI 3D annotations.
        # 2. cam2_projection_rectified (P2): Projects the rectified 3D coordinates onto the 2D pixel plane of Camera 2 (RGB camera).
        # Note: Even though KITTI 3D space is referenced to Cam0, visual tasks and YOLO models 
        #       typically use image_02 (Cam2 RGB images). This combined matrix ensures correct alignment and projection.
        cam0_to_cam2_projection = cam2_projection_rectified @ cam0_rectification

        print("=> Initializing dataset with accurate 3D-to-2D projection matching...")
        self.yolo = YOLO(yolo_model_path)
        
        self.cached_data = []
        for img_file in self.img_files:
            frame_idx = int(img_file.split('.')[0])
            img_path = os.path.join(self.img_dir, img_file)
            bin_path = os.path.join(self.bin_dir, f"{frame_idx:010d}.bin")
            
            if not os.path.exists(bin_path):
                continue
                
            img = cv2.imread(img_path)
            results = self.yolo(img, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            
            # Load point cloud once per frame to process all proposals efficiently
            point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
            pts_3d = point_cloud[:, :3]
            pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
            pts_cam = (self.ext @ pts_homo.T).T[:, :3]
            valid_mask = pts_cam[:, 2] > 0.1
            
            # Project points to 2D image plane to match YOLO proposal box
            pts_2d_homo = (self.proj @ pts_homo[valid_mask].T).T
            u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
            v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
            
            # print for dubug
            # print("image:", img_path)
            # print("box:", boxes)
            # print("clss:", clss)
            
            for box2d, cls_id in zip(boxes, clss):
                if int(cls_id) in TARGET_CLASSES.values():
                    u1, v1, u2, v2 = box2d
                    
                    valid_indices = np.where(valid_mask)[0]
                    box_mask = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
                    frustum_indices = valid_indices[box_mask]
                    frustum_pts = pts_3d[frustum_indices]
                    
                    if len(frustum_pts) <= 5:
                        continue
                        
                    # Compute the 2D center point of the YOLO box
                    yolo_center_u = (u1 + u2) / 2.0
                    yolo_center_v = (v1 + v2) / 2.0
                    
                    # Find the true matching object for each YOLO box 
                    # by projecting the 3D annotation back to 2D and measuring center distance.
                    best_p = None
                    best_tracklet = None
                    min_center_dist = float('inf')

                    for tracklet in self.annotations:
                        
                        if len(tracklet['poses']) == 0:
                            continue
                        
                        pose_idx = frame_idx - tracklet['first_frame']
                        if 0 <= pose_idx < len(tracklet['poses']):
                            p = tracklet['poses'][pose_idx]
                            
                            # Project the 3D annotation center onto the 2D image plane
                            gt_center_3d = np.array([p['tx'], p['ty'], p['tz'], 1.0])
                            cam_pt = self.ext @ gt_center_3d
                            if cam_pt[2] <= 0:
                                continue
                            
                            # Project the 3D point from the Camera 0 coordinate frame onto the Camera 2 (RGB image) 2D pixel plane
                            proj_pt = cam0_to_cam2_projection @ cam_pt
                            
                            if proj_pt[2] <= 1e-5:
                                continue
        
                            gt_u = proj_pt[0] / proj_pt[2]
                            gt_v = proj_pt[1] / proj_pt[2]
                            
                            # print for debug
                            # print(f"GT 3D Center Projected -> u: {gt_u:.1f}, v: {gt_v:.1f}")
                            # print(f"YOLO Box -> u1:{u1}, v1:{v1}, u2:{u2}, v2:{v2}")

                            # Calculate the projected center distance and check if it falls within the YOLO box
                            dist = np.sqrt((yolo_center_u - gt_u)**2 + (yolo_center_v - gt_v)**2)
                                    
                            # If the GT projection falls inside the YOLO box bounds and is closer, update the best match
                            # Keep track of the closest matching tracklet inside the YOLO box
                            if (u1 <= gt_u <= u2) and (v1 <= gt_v <= v2):
                                if dist < min_center_dist:
                                    min_center_dist = dist
                                    best_p = p
                                    best_tracklet = tracklet
                                    
                    if best_p is not None:
                        
                        # Build and save the final 3D box only after checking all tracklets to prevent loop overwriting
                        best_match_gt = np.array([
                            best_p['tx'], best_p['ty'], best_p['tz'], 
                            best_tracklet['l'], best_tracklet['w'], best_tracklet['h'], 
                            best_p['rz']
                        ], dtype=np.float32)
                        
                        # print("best_match:", best_match_gt)
                        
                        self.cached_data.append({
                            'img_path': img_path,
                            'bin_path': bin_path,
                            'frustum_pts': frustum_pts,
                            'gt_box': best_match_gt
                        })
                        
        print(f"=> Pre-caching complete with accurate matching. Total valid samples: {len(self.cached_data)}")

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        sample = self.cached_data[idx]
        frustum_pts = sample['frustum_pts']
        gt_box = sample['gt_box'].copy()
        
        centroid = np.mean(frustum_pts, axis=0)
        norm_pts = frustum_pts - centroid
        
        if len(norm_pts) >= 512:
            choice = np.random.choice(len(norm_pts), 512, replace=False)
        else:
            choice = np.random.choice(len(norm_pts), 512, replace=True)
        sampled_pts = norm_pts[choice]
        
        gt_box[:3] -= centroid
        
        return torch.tensor(sampled_pts, dtype=torch.float32), torch.tensor(gt_box, dtype=torch.float32)

if __name__ == "__main__":
    VISUALIZE_DATASET = True
    
    # 1. Automatically select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    dataset = KittiFrustumDataset(data_path, "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    
    if VISUALIZE_DATASET:
        print(f"=> Starting dataset visualization validation. Total samples: {len(dataset)}")

        for idx in range(len(dataset)):
            # Retrieve tensor data
            sampled_pts, gt_box = dataset[idx]
            
            # Retrieve original paths and data from cache for visualization
            raw_sample = dataset.cached_data[idx]
            
            print(f"Viewing sample {idx + 1} / {len(dataset)}...")
            
            # Call the Open3D visualization function
            # visualize_sample reads raw_sample['bin_path'] and raw_sample['img_path'], then draws the 3D bounding box using gt_box
            visualize_sample(
                bin_path=raw_sample['bin_path'], 
                img_path=raw_sample['img_path'], 
                gt_box=raw_sample['gt_box'] # Recommended to use either the raw gt_box or the centroid-aligned gt_box for inspection
            )
            
    # Split dataset into train (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 2. Move model to GPU
    model = SimpleFrustumPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/frustum_pointnet_experiment')
    
    # Enable Automatic Mixed Precision (AMP) for faster FP16 training and lower memory overhead
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    num_epochs = 100
    
    # Configure Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Training Loop across multiple epochs
    print(f"=> Starting training for {num_epochs} epochs across {len(train_loader)} training samples...")
    for epoch in range(num_epochs):
        # train
        model.train()
        total_train_loss = 0.0
        
        for batch_points, batch_gt_boxes in train_loader:
            # 3. Move batch data tensors to GPU
            batch_points = batch_points.to(device, non_blocking=True)
            batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
            
            optimizer.zero_grad()                                   # Clear previous gradients
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                predictions = model(batch_points)                       # Forward pass
                loss = F.smooth_l1_loss(predictions, batch_gt_boxes)    # Compute Smooth L1 Loss
            
            # Scaled backward pass
            scaler.scale(loss).backward()                          # Backward pass (compute gradients)
            scaler.step(optimizer)                                 # Update model weights
            scaler.update()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Eval
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_points, batch_gt_boxes in val_loader:
                batch_points = batch_points.to(device, non_blocking=True)
                batch_gt_boxes = batch_gt_boxes.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_points)
                    val_loss = F.smooth_l1_loss(predictions, batch_gt_boxes)
                    
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")        
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save comprehensive checkpoint dictionary
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, 'frustum_pointnet_checkpoint.pth')
            print("=> Saved best training checkpoint.")