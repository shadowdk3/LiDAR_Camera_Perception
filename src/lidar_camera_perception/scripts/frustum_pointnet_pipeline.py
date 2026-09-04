"""
End-to-End Frustum PointNet 3D Object Detection Pipeline
- Integrates YOLO 2D object detection with KITTI calibration matrices (LiDAR-to-Camera projection).
- Parses KITTI XML tracklet labels for ground truth 3D bounding box retrieval.
- Crops frustum point clouds, applies local zero-centering normalization, and handles batch tensor padding.
- Implements a PyTorch-native SimpleFrustumPointNet (Shared MLP + Max Pooling) for 3D bounding box regression [x, y, z, l, w, h, yaw].
- Computes smooth L1 training loss and renders both OpenCV 2D projections and Open3D interactive visualizations.
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
        # 安全取得 objectType
        obj_type_node = item.find('objectType')
        obj_type = obj_type_node.text if obj_type_node is not None else 'Unknown'
        
        # 安全取得尺寸數據
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

def draw_kitti_gt_boxes(img, gt_annotations, current_frame_idx, velo_to_cam2_proj):
    """Projects and draws KITTI 3D ground truth bounding boxes onto the camera image."""
    for gt_annotation in gt_annotations:
        start_frame = gt_annotation['first_frame']
        pose_idx = current_frame_idx - start_frame

        if 0 <= pose_idx < len(gt_annotation['poses']):
            pose = gt_annotation['poses'][pose_idx]
            h, w, l = gt_annotation['h'], gt_annotation['w'], gt_annotation['l']
            tx, ty, tz, rz = pose['tx'], pose['ty'], pose['tz'], pose['rz']
                        
            # Create 3D bounding box corners centered at the object origin
            x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            z_corners = [0, 0, 0, 0, h, h, h, h]
            corners = np.vstack([x_corners, y_corners, z_corners])
                        
            # Apply yaw rotation around the z-axis and translation to LiDAR/world coordinates            
            c, s = np.cos(rz), np.sin(rz)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            corners = R @ corners
            corners[0, :] += tx
            corners[1, :] += ty
            corners[2, :] += tz
                        
            # Project 3D bounding box corners onto the 2D image plane using the projection matrix
            ones = np.ones((1, corners.shape[1]))
            corners_homo = np.vstack((corners, ones))
            img_coords = velo_to_cam2_proj @ corners_homo
            x = img_coords[0, :] / img_coords[2, :]
            y = img_coords[1, :] / img_coords[2, :]
            pts_2d = np.vstack((x, y)).T.astype(int)
                        
            # Draw the 12 edges of the 3D bounding box
            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
            for start, end in lines:
                cv2.line(img, tuple(pts_2d[start]), tuple(pts_2d[end]), (0, 255, 0), 2)
                                
    # Display the final image with projected ground truth boxes and wait for a key press
    cv2.imshow("KITTI Tracklets Python Viewer", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
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
    label_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync/tracklet_labels.xml"
    
    img = cv2.imread(img_path)
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    gt_annotations = parse_kitti_tracklets(label_path)

    velo_to_cam2_proj, velo_to_cam0_ext = read_kitti_calib()
    yolo_model = YOLO("/home/user/LiDAR_Camera_Perception_ws/yolo11n.pt") 
    results = yolo_model(img)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    
    frustum_list = []
    all_frustums_pcd = []
    gt_boxes_list = []
    
    print(f"Detected {len(boxes)} 2D objects, starting pipeline...")
    
    # Extract current frame index from image filename
    current_frame_idx = int(os.path.basename(img_path).split('.')[0])
    print("current_frame_idx", current_frame_idx)
    
    for i, (box2d, cls_id) in enumerate(zip(boxes, clss)):
        if int(cls_id) in TARGET_CLASSES.values():
            frustum_pts = crop_frustum_points(point_cloud, box2d, velo_to_cam2_proj, velo_to_cam0_ext)
            
            if len(frustum_pts) > 5:
                # Step 1: Zero-centering (Normalization)
                centroid = np.mean(frustum_pts, axis=0)
                norm_pts = frustum_pts - centroid
                frustum_list.append(norm_pts)
                    
                # Step 2: Find and match the corresponding ground truth (GT) 3D bounding box for the current frame                matched_gt_box = None
                min_dist = float('inf')
                
                for tracklet in gt_annotations:
                    start_frame = tracklet['first_frame']
                    pose_idx = current_frame_idx - start_frame
                    
                    if 0 <= pose_idx < len(tracklet['poses']):
                        pose = tracklet['poses'][pose_idx]
                        h, w, l = tracklet['h'], tracklet['w'], tracklet['h']
                        tx, ty, tz, rz = pose['tx'], pose['ty'], pose['tz'], pose['rz']
                        
                        # Simple matching strategy: project the GT 3D center back to 2D and compute distance or IoU against the current YOLO box2d
                        # Here, we use 2D image center point distance to find the nearest GT as a matching example
                        gt_center_3d = np.array([tx, ty, tz, 1.0])
                        gt_center_2d = velo_to_cam2_proj @ (velo_to_cam0_ext @ gt_center_3d) if 'velo_to_cam0_ext' in locals() else velo_to_cam2_proj @ gt_center_3d
                        # (Alternative: directly check if the GT falls within the YOLO 2D bounding box)                        
                        
                        box_u1, box_v1, box_u2, box_v2 = box2d
                        # Construct the GT vector for this frame [x, y, z, l, w, h, yaw]
                        gt_box = np.array([tx, ty, tz, tracklet['l'], tracklet['w'], tracklet['h'], rz])
                        
                        # In real-world projects, 2D/3D IoU is typically used for strict matching; here we grab valid candidates
                        matched_gt_box = gt_box
                        break # Break after finding (recommended to add IoU checks in practice to prevent wrong targets)
                    
                if matched_gt_box is not None:
                    # Step 3: Subtract the exact same centroid from the GT center coordinates to shift it into the local coordinate system
                    gt_box_aligned = matched_gt_box.copy()
                    gt_box_aligned[:3] -= centroid 
                    gt_boxes_list.append(gt_box_aligned)
                    
                # Open3D Visualization setup
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frustum_pts)
                pcd.paint_uniform_color(np.random.rand(3))
                all_frustums_pcd.append(pcd)

    # This is currently just an inference / evaluation pass to calculate the target loss
    # After exiting the YOLO detection and matching loops, proceed with tensor conversion and model loss calculation
    # Tensor conversion, batch padding, model inference, and loss calculation
    if len(frustum_list) > 0 and len(gt_boxes_list) == len(frustum_list):
        # Assuming your model requires a fixed number of points (e.g., downsampling or padding each frustum to 512 points)
        # Here we demonstrate reshaping each frustum to a uniform shape (Batch, 512, 3)
        batch_points = []
        for pts in frustum_list:
            if len(pts) >= 512:
                choice = np.random.choice(len(pts), 512, replace=False)
            else:
                choice = np.random.choice(len(pts), 512, replace=True)
            batch_points.append(pts[choice])
            
        tensor_input_points = torch.tensor(np.array(batch_points), dtype=torch.float32) 
        tensor_gt_boxes = torch.tensor(np.array(gt_boxes_list), dtype=torch.float32) 
        
        # Execute model inference and loss calculation
        model = SimpleFrustumPointNet()
        model.eval()
        with torch.no_grad():
            predictions = model(tensor_input_points)
        
        loss = F.smooth_l1_loss(predictions, tensor_gt_boxes)
        print("Training Loss computed successfully:", loss.item())
        
    # Show the 3D Ground Truth (GT) bounding boxes on the camera image
    draw_kitti_gt_boxes(img, gt_annotations, current_frame_idx, velo_to_cam2_proj)
                
    # Background Point Cloud Visualization
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    bg_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    o3d.visualization.draw_geometries([bg_pcd, *all_frustums_pcd], 
                                        window_name="Frustum Point Cloud & PointNet Test")
    