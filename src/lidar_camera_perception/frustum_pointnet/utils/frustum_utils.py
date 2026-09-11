import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import xml.etree.ElementTree as ET
import open3d as o3d
from torch.utils.data import Dataset

from ultralytics import YOLO
from shapely.geometry import Polygon

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

# SimpleFrustumPointNet: A PyTorch implementation that processes a cropped 3D point cloud 
# frustum to extract point-wise and global features, ultimately regressing a 3D bounding 
# box parameterized by [x, y, z, l, w, h, yaw].
class SimpleFrustumPointNet(nn.Module):
    def __init__(self):
        super(SimpleFrustumPointNet, self).__init__()
        # Shared MLP (Multilayer Perceptron) for pointwise feature extraction
        # Transforms each point's 3D coordinates (x, y, z) into a rich 1024-dimensional feature vector
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
        
        # Regression Head: Maps the global shape descriptor to 3D bounding box parameters
        # Output: [x, y, z, l, w, h, yaw]
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, points):
        # Rearrange input point cloud shape from (Batch_Size, Num_Points, 3) to (Batch_Size, 3, Num_Points)
        # Required format for PyTorch 1D convolutions (treating channels as spatial dimensions)
        x = points.permute(0, 2, 1)
        
        # Extract high-dimensional features for every individual point -> Output shape: (B, 1024, N)
        x = self.mlp1(x)
        
        # Apply symmetric max-pooling across all points (dim=2) to aggregate a global feature vector
        # This ensures the network is invariant to the ordering of points -> Output shape: (B, 1024)
        x = torch.max(x, dim=2)[0] # (B, 1024)
        
        # Regress final 3D bounding box parameters from the global feature vector -> Output shape: (B, 7)
        box_params = self.fc(x) # (B, 7)
        return box_params

class FrustumPointNetV2(nn.Module):
    def __init__(self):
        super(FrustumPointNetV2, self).__init__()
        
        # T-Net: Predicts a 3D translation offset to center the point cloud
        self.tnet = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3) # outputs [dx, dy, dz]
        )

        # Shared MLP for point feature extraction (after centering)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        
        # Segmentation Head: Classifies each point as foreground (object) or background
        # Output: (B, 2, N) for binary classification per point
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 2, 1) 
        )

        # Feature extraction for masked foreground points
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        
        # Regression Head for 3D Bounding Box: [x, y, z, l, w, h, yaw]
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, points):
        # points shape: (B, N, 3)
        B, N, _ = points.shape
        x = points.permute(0, 2, 1) # (B, 3, N)
        
        # 1. Spatial alignment using T-Net
        center_offset = self.tnet(x) # (B, 3)
        x_centered = x - center_offset.unsqueeze(2) # Center the points
        
        # 2. Extract local point features
        point_features = self.mlp1(x_centered) # (B, 128, N)
        
        # 3. Predict point-wise segmentation masks (Foreground vs Background)
        logits = self.seg_head(point_features) # (B, 2, N)
        mask = torch.argmax(logits, dim=1) # (B, N) - 1 for foreground, 0 for background
        
        # 4. Enhance features using only foreground points (or weight them)
        # Pass through second feature extractor
        global_features = self.mlp2(point_features) # (B, 1024, N)
        
        # Apply mask during max-pooling to ignore background noise
        # Expand mask to match feature channels: (B, 1, N)
        mask_expanded = mask.unsqueeze(1).float()
        global_features = global_features * mask_expanded
        
        # Symmetric max pooling across points
        x_global = torch.max(global_features, dim=2)[0] # (B, 1024)
        
        # 5. Regress final box parameters
        box_params = self.fc(x_global) # (B, 7)
        
        # Add back the T-Net offset to spatial coordinates [x, y, z]
        box_params[:, 0:3] += center_offset
        
        return box_params, logits, center_offset
    
# Custom3DCornerLoss: A PyTorch loss module that computes the distance between predicted 
# and ground-truth 3D bounding boxes by converting 7D box parameters ([x, y, z, l, w, h, rz]) 
# into their 8 absolute 3D corner coordinates and evaluating their average absolute error.
class Custom3DCornerLoss(nn.Module):
    def __init__(self):
        super(Custom3DCornerLoss, self).__init__()
        # Optional loss function helper (currently using direct L1 mean in forward)
        self.smooth_l1 = nn.SmoothL1Loss()

    def get_box_corners(self, box_params):
        # Unpack 7D bounding box parameters: center [x, y, z], dimensions [l, w, h], and yaw rotation [rz]
        x, y, z = box_params[:, 0], box_params[:, 1], box_params[:, 2]
        l, w, h = box_params[:, 3], box_params[:, 4], box_params[:, 5]
        rz = box_params[:, 6]

        batch_size = box_params.shape[0]
        device = box_params.device

        # 1. Generate the standard 8 corner coordinates relative to the box center (unrotated)
        x_corners = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1)
        y_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)
        z_corners = torch.stack([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1)
        corners = torch.stack([x_corners, y_corners, z_corners], dim=2) # [B, 8, 3]

        # 2. Construct the 3x3 Z-axis rotation matrix using the yaw angle (rz)
        c, s = torch.cos(rz), torch.sin(rz)
        zeros = torch.zeros_like(c)
        ones = torch.ones_like(c)
        
        rot_matrix = torch.stack([
            c, -s, zeros,
            s,  c, zeros,
            zeros, zeros, ones
        ], dim=1).reshape(batch_size, 3, 3)

        # 3. Apply rotation to the corners and translate them to absolute 3D world coordinates        rotated_corners = torch.matmul(corners, rot_matrix.transpose(1, 2))
        rotated_corners = torch.matmul(corners, rot_matrix.transpose(1, 2))
        center = torch.stack([x, y, z], dim=1).unsqueeze(1) # [B, 1, 3]
        abs_corners = rotated_corners + center

        return abs_corners

    def forward(self, pred_boxes, gt_boxes):
        # Obtain absolute 3D corner coordinates for both predicted and ground-truth boxes
        pred_corners = self.get_box_corners(pred_boxes)
        gt_corners = self.get_box_corners(gt_boxes)
       
        # Calculate the mean absolute error (L1 loss) across all 8 corner points
        loss = torch.mean(torch.abs(pred_corners - gt_corners))
        return loss
    
# read_kitti_calib: Returns standard KITTI dataset calibration matrices (LiDAR-to-camera 
# extrinsic, camera rectification, and left color camera projection) and computes composite 
# transformation matrices to project 3D LiDAR points directly onto image plane pixels.
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

    # Compute composite transformation matrices via matrix multiplication
    # Maps 3D LiDAR point cloud coordinates directly to 2D Cam2 pixel coordinates
    velo_to_cam2_projection = cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_extrinsic
    
    # it maps a 3D point in the Camera 0 coordinate system directly to 2D pixel coordinates on the Left Color Camera (Cam2) image.
    cam0_to_cam2_projection = cam2_projection_rectified @ cam0_rectification
    
    return velo_to_cam2_projection, velo_to_cam0_extrinsic, cam0_to_cam2_projection

# parse_kitti_tracklets: Parses a KITTI dataset tracklet XML file to extract 
# 3D bounding box dimensions, object types, starting frames, and time-varying 
# poses (translations and yaw rotations) across sequence frames.
def parse_kitti_tracklets(xml_path):
    # Load and parse the XML tracklet file structure
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets = []
    
    # Iterate through each object tracklet item entry in the XML
    for item in root.findall('.//item'):
        obj_type_node = item.find('objectType')
        obj_type = obj_type_node.text if obj_type_node is not None else 'Unknown'
        
        # Extract object physical dimensions (height, width, length) and initial frame index
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
        
        # Extract sequential 3D trajectory poses (spatial translations and yaw rotation)
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

# visualize_sample: Loads a KITTI LiDAR point cloud and an RGB image, builds an Open3D 
# point cloud object, creates oriented 3D bounding boxes for ground-truth (green) and 
# optional predictions (red), and renders them in an interactive 3D visualization window.
def visualize_sample(bin_path, gt_box, pred_box=None):
    # 1. Load the image and raw LiDAR point cloud data
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
    
# visualize_sample: Loads a KITTI LiDAR point cloud and an RGB image, builds an Open3D 
# point cloud object, creates oriented 3D bounding boxes for ground-truth (green) and 
# optional predictions (red), and renders them in an interactive 3D visualization window.
def visualize_all_sample(bin_path, gt_boxes, pred_boxes=None):
    # 1. Load the image and raw LiDAR point cloud data
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
    
    # Handle multiple ground-truth boxes (iterable list or single box)
    if isinstance(gt_boxes, np.ndarray) and gt_boxes.ndim == 1:
        gt_boxes = [gt_boxes]
    for gt_box in gt_boxes:
        geometries.append(create_o3d_box(gt_box, color=[0, 1, 0]))
    
    # Handle multiple prediction boxes
    if pred_boxes is not None:
        if isinstance(pred_boxes, np.ndarray) and pred_boxes.ndim == 1:
            pred_boxes = [pred_boxes]
        for pred_box in pred_boxes:
            geometries.append(create_o3d_box(pred_box, color=[1, 0, 0]))
        
    # 4. Open an interactive 3D visualization window (supports mouse rotation, translation, and zoom)
    print("=> Displaying 3D point cloud and bounding boxes (use mouse to rotate, zoom, and pan)")
    o3d.visualization.draw_geometries(geometries)
    
# RAM Caching & Data Pipeline Optimization
# Pre-Generation (__init__): Runs YOLO once per frame beforehand to index every bounding box proposal separately, 
#     avoiding CUDA multi-threading crashes and capturing multiple objects per image.
# Frustum Extraction (__getitem__): Keeps your exact point-cloud projection, 2D box filtering, zero-center normalization, 
#     and 512-point uniform sampling logic intact.
# DataLoader Usage: When instantiating your DataLoader, explicitly set `num_workers=0` to ensure safe interaction
#     with YOLO model weights.
class KittiFrustumDataset(Dataset):
    def __init__(self, data_dir, yolo_model_path):
        self.img_dir = os.path.join(data_dir, "image_00/data")
        self.bin_dir = os.path.join(data_dir, "velodyne_points/data")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.annotations = parse_kitti_tracklets(os.path.join(data_dir, "tracklet_labels_cleaned.xml"))
        
        # Construct a composite matrix to project from unrectified Camera 0 coordinates to Camera 2 (RGB image) pixels.
        # 1. cam0_rectification (R0_rect): Rectifies the raw Camera 0 coordinate system 
        #    to align with the standard rectified coordinate system used by KITTI 3D annotations.
        # 2. cam2_projection_rectified (P2): Projects the rectified 3D coordinates onto the 2D pixel plane of Camera 2 (RGB camera).
        # Note: Even though KITTI 3D space is referenced to Cam0, visual tasks and YOLO models 
        #       typically use image_02 (Cam2 RGB images). This combined matrix ensures correct alignment and projection.
        self.velo_cam2_projection, self.velo_to_cam0_extrinsic, self.cam0_to_cam2_projection = read_kitti_calib()

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
            pts_cam = (self.velo_to_cam0_extrinsic @ pts_homo.T).T[:, :3]
            valid_mask = pts_cam[:, 2] > 0.1
            
            # Project points to 2D image plane to match YOLO proposal box
            pts_2d_homo = (self.velo_cam2_projection @ pts_homo[valid_mask].T).T
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
                    
                    pts_cam_in_box = pts_cam[valid_mask][box_mask]
                    
                    if len(pts_cam_in_box) > 30:
                        # Filter out ground points and background outliers during frustum extraction
                        y_cam = pts_cam_in_box[:, 1]
                        y_min, y_max = np.percentile(y_cam, 5), np.percentile(y_cam, 90)
                        clean_mask = (y_cam >= y_min) & (y_cam <= y_max)
                        frustum_pts = frustum_pts[clean_mask]
                    
                    if len(frustum_pts) < 5:
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
                            cam_pt = self.velo_to_cam0_extrinsic @ gt_center_3d
                            if cam_pt[2] <= 0:
                                continue
                            
                            # Project the 3D point from the Camera 0 coordinate frame onto the Camera 2 (RGB image) 2D pixel plane
                            proj_pt = self.cam0_to_cam2_projection @ cam_pt
                            
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

    # Empty Frustum Check: Returns zero-tensors if no LiDAR points are found.
    # Local Masking: Transforms points into the box's local coordinate frame via inverse Z-axis rotation to generate binary foreground/background segmentation masks.
    # Centroid Normalization: Shifts the point cloud to center it at the origin.
    # Uniform Sampling: Randomly samples exactly 512 points using identical index tracking to keep points and masks perfectly aligned.
    # GT Box Alignment: Adjusts the 3D bounding box center coordinates by subtracting the point cloud centroid.
    # Tensor Conversion: Outputs final PyTorch tensors (float32 and long) ready for training ingestion.
    def __getitem__(self, idx):
        sample = self.cached_data[idx]
        frustum_pts = sample['frustum_pts']
        gt_box = sample['gt_box'].copy()
        
        # Calculate segmentation mask (0: background/other, 1: target object) within the 3D bounding box
        cx, cy, cz, l, w, h, rz = gt_box
        pts_trans = frustum_pts - np.array([cx, cy, cz])
        
        # Transform points into the box's local coordinate system via inverse rotation around the Z-axis (-rz)
        cos_r = np.cos(-rz)
        sin_r = np.sin(-rz)
        x_local = cos_r * pts_trans[:, 0] - sin_r * pts_trans[:, 1]
        y_local = sin_r * pts_trans[:, 0] + cos_r * pts_trans[:, 1]
        z_local = pts_trans[:, 2]
        
        # Determine whether each point lies strictly within the 3D bounding box boundaries
        mask = (np.abs(x_local) <= l / 2.0) & \
               (np.abs(y_local) <= w / 2.0) & \
               (np.abs(z_local) <= h / 2.0)
        mask = mask.astype(np.int64)
        
        # Perform centroid normalization on the frustum point cloud
        centroid = np.mean(frustum_pts, axis=0)
        norm_pts = frustum_pts - centroid
        
        # Fixed sampling of 512 points (applying identical random indices to both points and mask)
        num_pts = len(norm_pts)
        if num_pts >= 512:
            choice = np.random.choice(num_pts, 512, replace=False)
        else:
            choice = np.random.choice(num_pts, 512, replace=True)
            
        sampled_pts = norm_pts[choice]
        sampled_mask = mask[choice]
        
        # Synchronize the GT box center by subtracting the point cloud centroid
        gt_box[:3] -= centroid
        
        return (
            torch.tensor(sampled_pts, dtype=torch.float32), 
            torch.tensor(gt_box, dtype=torch.float32), 
            torch.tensor(sampled_mask, dtype=torch.long)
        )

# draw_3d_box: Projects a 3D bounding box defined by 7D parameters onto a 2D image plane 
# using a projection matrix, handling coordinate transformations and rendering the 12 edges 
# of the bounding box using OpenCV.
def draw_3d_box(img, box_params, proj_matrix, color):
    # Unpack 7 parameters: [x, y, z, l, w, h, yaw]
    x, y, z, l, w, h, rz = box_params
    
    # Create 3D bounding box corners centered at origin
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Apply yaw rotation and translation
    c, s = np.cos(rz), np.sin(rz)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    # Project onto 2D image plane
    ones = np.ones((1, corners.shape[1]))
    corners_homo = np.vstack((corners, ones))
    img_coords = proj_matrix @ corners_homo
    
    if np.any(img_coords[2, :] <= 0):
        return img # Skip boxes behind camera
        
    px = img_coords[0, :] / img_coords[2, :]
    py = img_coords[1, :] / img_coords[2, :]
    pts_2d = np.vstack((px, py)).T.astype(int)
    
    # Draw 12 edges of the bounding box
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for start, end in lines:
        cv2.line(img, tuple(pts_2d[start]), tuple(pts_2d[end]), color, 2)
    return img


# Computes the Bird's-Eye View (BEV) IoU between two 3D boxes.
# Box format: [x, y, z, l, w, h, rz] (KITTI coordinate system)
def compute_bev_iou(box1, box2):
  
    def get_corners(box):
        x, y, z, l, w, h, rz = box
        # Correct BEV 4-corner order (front-left, front-right, rear-right, rear-left)
        x_corners = [l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2]
        corners = np.vstack([x_corners, y_corners])
        
        # Rotation matrix (around Z-axis)
        c, s = np.cos(rz), np.sin(rz)
        R = np.array([[c, -s], [s, c]])
        corners = np.dot(R, corners)
        
        # Translate to actual center
        corners[0, :] += x
        corners[1, :] += y
        return corners.T # Transpose to shape (4, 2)
    
    try:
        poly1 = Polygon(get_corners(box1))
        poly2 = Polygon(get_corners(box2))
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    except Exception:
        return 0.0

def nms_3d(boxes, scores, iou_threshold=0.1):
    """
    3D NMS filtering function.
    boxes: Tensor or numpy array, shape [N, 7] -> [x, y, z, l, w, h, rz] scores:
    Tensor or numpy array, shape [N] -> predicted confidence scores
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
        
    if len(boxes) == 0:
        return []

    # Sort by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    keep = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
            
        current_box = boxes[current]
        other_boxes = boxes[sorted_indices[1:]]
        other_indices = sorted_indices[1:]
        
        # Force compute and print IoU for each box pair
        for idx, b in zip(other_indices, other_boxes):
            iou = compute_bev_iou(current_box, b)
            print(f"Comparing Box {current} and Box {idx} -> BEV IoU: {iou:.4f}")
            
        # Compute BEV IoU between current box and remaining boxes
        ious = np.array([compute_bev_iou(current_box, b) for b in other_boxes])
        
        # Keep items with IoU less than the threshold (filtering out redundant boxes with high overlap)        
        valid_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[valid_indices + 1]

    return keep