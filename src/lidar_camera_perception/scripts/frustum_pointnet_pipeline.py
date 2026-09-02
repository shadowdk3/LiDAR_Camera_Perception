"""
Frustum-to-Voxel 3D Object Detection Pipeline
- Integrates YOLO 2D object detection with KITTI LiDAR calibration.
- Crops frustum point clouds, applies zero-centering normalization, and tensor batch padding.
- Utilizes a PyTorch-native SimpleFrustumPointNet for 3D bounding box regression [x, y, z, l, w, h, yaw].
"""

import numpy as np
import cv2
import open3d as o3d
import torch
import torch.nn as nn
from ultralytics import YOLO

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
        # Shared MLP (PointNet Core Feature Extraction)
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

def crop_frustum_points(point_cloud, box2d, velo_to_cam2_proj, velo_to_cam0_ext):
    pts_3d = point_cloud[:, :3]
    pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    
    pts_cam = (velo_to_cam0_ext @ pts_homo.T).T[:, :3]
    valid_mask = pts_cam[:, 2] > 0.1
    
    pts_homo_valid = pts_homo[valid_mask]
    pts_3d_valid = pts_3d[valid_mask]
    
    pts_2d_homo = (velo_to_cam2_proj @ pts_homo_valid.T).T
    u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
    v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
    
    u1, v1, u2, v2 = box2d
    in_box_mask = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
    
    return pts_3d_valid[in_box_mask]

if __name__ == "__main__":
    img_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync/image_00/data/0000000072.png"
    bin_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/0000000072.bin"
    
    img = cv2.imread(img_path)
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    velo_to_cam2_proj, velo_to_cam0_ext = read_kitti_calib()
    yolo_model = YOLO("/home/user/LiDAR_Camera_Perception_ws/yolo11n.pt") 
    results = yolo_model(img)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    
    frustum_list = []
    all_frustums_pcd = []
    
    print(f"Detected {len(boxes)} 2D objects, starting pipeline...")
    
    for i, (box2d, cls_id) in enumerate(zip(boxes, clss)):
        if int(cls_id) in TARGET_CLASSES.values():
            frustum_pts = crop_frustum_points(point_cloud, box2d, velo_to_cam2_proj, velo_to_cam0_ext)
            
            if len(frustum_pts) > 5:
                # Step 1: Zero-centering (Normalization)
                centroid = np.mean(frustum_pts, axis=0)
                norm_pts = frustum_pts - centroid
                frustum_list.append(norm_pts)
                
                # Open3D Visualization setup
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frustum_pts)
                pcd.paint_uniform_color(np.random.rand(3))
                all_frustums_pcd.append(pcd)

    # Step 2: Batch Padding & Tensor Alignment
    max_points = 512
    batch_size = len(frustum_list)
    
    if batch_size > 0:
        batch_points = np.zeros((batch_size, max_points, 3), dtype=np.float32)
        for i, pts in enumerate(frustum_list):
            num_pts = len(pts)
            if num_pts >= max_points:
                choices = np.random.choice(num_pts, max_points, replace=False)
                batch_points[i] = pts[choices]
            else:
                batch_points[i, :num_pts, :] = pts
                
        tensor_input_points = torch.tensor(batch_points)
        print("Model Input Tensor Shape (Batch, N, 3):", tensor_input_points.shape)
        
        # Step 3: PointNet Inference Test
        model = SimpleFrustumPointNet()
        model.eval()
        with torch.no_grad():
            predictions = model(tensor_input_points)
        print("Model Prediction Output Shape (Batch, 7):", predictions.shape)

    # Background Point Cloud Visualization
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    bg_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    o3d.visualization.draw_geometries([bg_pcd, *all_frustums_pcd], 
                                        window_name="Frustum Point Cloud & PointNet Test")
    