"""
This script reads 2D YOLO detection results, LiDAR point clouds, and KITTI calibration matrices (Calib). 
It projects the point cloud onto a 2D image for cropping, and finally uses Open3D to visualize 
the extracted 3D frustum point clouds.

Step 1 (2D Detection): YOLO takes the 2D image and detects objects, outputting 2D bounding boxes [u_1, v_1, u_2, v_2].

Step 2 (3D Projection & Frustum Cropping): Using calibration matrices, all 3D LiDAR points are projected 
    into the image frame. Points that fall inside a YOLO 2D box are kept, creating a 3D frustum 
    (a pyramid-like 3D slice) of points for each detected object.

Step 3 (Coloring): Each isolated 3D frustum is assigned a distinct color (pcd.paint_uniform_color(...)) 
    to visually separate detected objects from the uncolored (grey) background point cloud.
"""

import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# for the detection result, only consider the following classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
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

# ==========================================
# Read KITTI Calibration Matrices
# ==========================================
def read_kitti_calib():
    """
    Construct KITTI Raw Calibration matrices and pre-compute the unified projection matrix.
    
    Returns:
        velo_to_cam2_projection: 3x4 unified projection matrix (LiDAR Frame -> Cam2 Pixel Frame)
        velo_to_cam0_extrinsic: 4x4 extrinsic matrix (LiDAR Frame -> Cam0 Frame, used to filter points with Z > 0 in front)
    """
    
    # 1. Extrinsic Matrix from LiDAR (Velodyne) to Cam0 (4x4)
    velo_to_cam0_extrinsic = np.array([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])

    # 2. Rectification Matrix for Cam0, R_rect_00 (4x4)
    cam0_rectification = np.array([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])

    # 3. Projection Matrix for Left Color Camera (Cam2), P_rect_02 (3x4)
    cam2_projection_rectified = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])

    # Unified 3D-to-2D Projection Matrix (3x4)
    velo_to_cam2_projection = cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_extrinsic

    return velo_to_cam2_projection, velo_to_cam0_extrinsic

# ==========================================
# Frustum Cropping
# ==========================================
def crop_frustum_points(point_cloud, box2d, velo_to_cam2_proj, velo_to_cam0_ext):
    """
    point_cloud: (N, 3) or (N, 4) LiDAR point cloud (X, Y, Z, [Reflectance])
    box2d: [u1, v1, u2, v2] 2D YOLO bounding box
    velo_to_cam2_proj: Pre-computed 3x4 projection matrix
    velo_to_cam0_ext: 4x4 LiDAR to Cam0 extrinsic matrix
    """
    pts_3d = point_cloud[:, :3]
    
    # Transform LiDAR points (Velodyne Frame) to Camera Frame
    pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1)))) # (N, 4)
    
    # Transform to camera coordinates and keep only points in front of the camera (Z > 0.1m)    
    pts_cam = (velo_to_cam0_ext @ pts_homo.T).T[:, :3] # (N, 3)
    valid_mask = pts_cam[:, 2] > 0.1
    
    pts_homo_valid = pts_homo[valid_mask]
    pts_3d_valid = pts_3d[valid_mask]
    
    # Project valid LiDAR points directly to 2D pixel homogeneous coordinates (u, v, z)
    pts_2d_homo = (velo_to_cam2_proj @ pts_homo_valid.T).T # (N, 3)
    
    # Normalize to get 2D pixel coordinates
    u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
    v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
    
    # Check if projected points fall within the YOLO 2D Box
    u1, v1, u2, v2 = box2d
    in_box_mask = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
    
    # Return the frustum point cloud in the LiDAR coordinate system
    frustum_pts_velo = pts_3d_valid[in_box_mask]
    return frustum_pts_velo

# ==========================================
# main
# ==========================================
if __name__ == "__main__":
    img_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync/image_00/data/0000000072.png"
    bin_path = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/0000000072.bin"
    
    # Load data
    img = cv2.imread(img_path)                                           # get image
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4) # KITTI is (N, 4)
    
    # Get pre-computed calibration matrices
    velo_to_cam2_proj, velo_to_cam0_ext = read_kitti_calib()
    
    # Run YOLO to get 2D detection results
    yolo_model = YOLO("/home/user/LiDAR_Camera_Perception_ws/yolo11n.pt") 
    results = yolo_model(img)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()  # Get [u1, v1, u2, v2]
    clss = results.boxes.cls.cpu().numpy()    # Class ID
    
    # Create a list for Open3D visualization objects
    all_frustums_pcd = []
    
    print(f"Detected {len(boxes)} 2D objects, starting Frustum cropping...")
    
    for i, (box2d, cls_id) in enumerate(zip(boxes, clss)):
        # Crop only for define class
        if int(cls_id) in TARGET_CLASSES.values():
            frustum_pts = crop_frustum_points(point_cloud, box2d, velo_to_cam2_proj, velo_to_cam0_ext)
            
            print(f"Object {i} [{yolo_model.names[int(cls_id)]}]: Extracted {len(frustum_pts)} 3D points")            
            
            if len(frustum_pts) > 0:
                # Convert to Open3D point cloud object and assign a random color for distinction                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frustum_pts)
                pcd.paint_uniform_color(np.random.rand(3))  # Random color
                all_frustums_pcd.append(pcd)

    # Draw the full point cloud as background (gray) and overlay cropped Frustum point clouds (colored)
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    bg_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Light gray background
    
    # Open 3D window for rendering
    o3d.visualization.draw_geometries([bg_pcd, *all_frustums_pcd], 
                                      window_name="Frustum Point Cloud Cropping")