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
    return velo_to_cam2_projection, velo_to_cam0_extrinsic

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
        self.annotations = parse_kitti_tracklets(os.path.join(data_dir, "tracklet_labels.xml"))
        self.proj, self.ext = read_kitti_calib()
        
        print("=> Initializing dataset and pre-generating YOLO 2D proposals...")
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
                    
            for box2d, cls_id in zip(boxes, clss):
                if int(cls_id) in TARGET_CLASSES.values():
                    u1, v1, u2, v2 = box2d
                    in_box = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
                    frustum_pts = pts_3d[valid_mask][in_box]
                    
                    if len(frustum_pts) < 5:
                        sampled_pts = np.zeros((512, 3), dtype=np.float32)
                        gt_box = np.zeros(7, dtype=np.float32)
                    else:
                        centroid = np.mean(frustum_pts, axis=0)
                        norm_pts = frustum_pts - centroid
                        
                        if len(norm_pts) >= 512:
                            choice = np.random.choice(len(norm_pts), 512, replace=False)
                        else:
                            choice = np.random.choice(len(norm_pts), 512, replace=True)
                        sampled_pts = norm_pts[choice]
                        
                        gt_box = np.zeros(7, dtype=np.float32)
                        for tracklet in self.annotations:
                            pose_idx = frame_idx - tracklet['first_frame']
                            if 0 <= pose_idx < len(tracklet['poses']):
                                p = tracklet['poses'][pose_idx]
                                gt_box = np.array([p['tx'], p['ty'], p['tz'], tracklet['l'], tracklet['w'], tracklet['h'], p['rz']], dtype=np.float32)
                                gt_box[:3] -= centroid  # Align with zero-centered frustum
                                break
                    
                    # Convert to tensors immediately and store in RAM cache list
                    self.cached_data.append((
                        torch.tensor(sampled_pts, dtype=torch.float32),
                        torch.tensor(gt_box, dtype=torch.float32)
                    ))
                    
            print(f"=> Successfully pre-cached {len(self.cached_data)} samples into RAM.")
            
    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        # Returns pre-loaded tensors instantly without disk reads or CPU math bottlenecks
        return self.cached_data[idx]
    
if __name__ == "__main__":
    # 1. Automatically select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    data_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    dataset = KittiFrustumDataset(data_path, "/home/user/LiDAR_Camera_Perception_ws/models/yolo11n.pt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    # 2. Move model to GPU
    model = SimpleFrustumPointNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/frustum_pointnet_experiment')
    
    # Enable Automatic Mixed Precision (AMP) for faster FP16 training and lower memory overhead
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    num_epochs = 30
    
    # Training Loop across multiple epochs
    print(f"=> Starting training for {num_epochs} epochs across {len(dataloader)} training samples...")
    for epoch in range(num_epochs):
        # train
        model.train()
        total_train_loss = 0.0
        
        for batch_points, batch_gt_boxes in dataloader:
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
            
        avg_loss = total_train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Log training loss to TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'frustum_pointnet.pth')
            print("=> Saved best model weights to frustum_pointnet.pth")