import os
import cv2
import torch
import onnxruntime as ort
import numpy as np

from pathlib import Path
from ultralytics import YOLO

from rosbags.rosbag2 import Reader
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent.parent
WS_DIR = SCRIPT_DIR.parent.parent
model_path = os.path.join(SCRIPT_DIR, "frustum_pointnet", "checkpoints", "frustum_pointnet_fine_tune.onnx")
bag_path = os.path.join(WS_DIR, "kitti_dataset_0009")
yolo_path = os.path.join(WS_DIR, "models", "yolo11n.pt")
    
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

def extract_frustum_data_from_pc(point_cloud, box2d):
    """
    Extracts frustum points from a raw point cloud and a 2D bounding box 
    using the exact calibration projection and filtering logic of KittiFrustumDataset.
    """
    velo_cam2_projection, velo_to_cam0_extrinsic, _ = read_kitti_calib()
    
    pts_3d = point_cloud[:, :3]
    pts_homo = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_cam = (velo_to_cam0_extrinsic @ pts_homo.T).T[:, :3]
    valid_mask = pts_cam[:, 2] > 0.1
    
    if not np.any(valid_mask):
        return None, None
        
    pts_2d_homo = (velo_cam2_projection @ pts_homo[valid_mask].T).T
    u = pts_2d_homo[:, 0] / pts_2d_homo[:, 2]
    v = pts_2d_homo[:, 1] / pts_2d_homo[:, 2]
    
    u1, v1, u2, v2 = box2d
    valid_indices = np.where(valid_mask)[0]
    box_mask = (u >= u1) & (u <= u2) & (v >= v1) & (v <= v2)
    frustum_indices = valid_indices[box_mask]
    frustum_pts = pts_3d[frustum_indices]
    
    pts_cam_in_box = pts_cam[valid_mask][box_mask]
    
    if len(pts_cam_in_box) > 30:
        y_cam = pts_cam_in_box[:, 1]
        y_min, y_max = np.percentile(y_cam, 5), np.percentile(y_cam, 90)
        clean_mask = (y_cam >= y_min) & (y_cam <= y_max)
        frustum_pts = frustum_pts[clean_mask]
        
    if len(frustum_pts) < 5:
        return None, None
        
    centroid = np.mean(frustum_pts, axis=0)
    return frustum_pts, centroid

def sample_and_normalize_points(frustum_pts, num_samples=512):
    """
    Normalizes frustum points relative to their centroid and 
    downsamples/pads them to a fixed count (default 512 points).
    """
    centroid = np.mean(frustum_pts, axis=0)
    norm_pts = frustum_pts - centroid
    
    num_pts = len(norm_pts)
    if num_pts >= num_samples:
        choice = np.random.choice(num_pts, num_samples, replace=False)
    else:
        choice = np.random.choice(num_pts, num_samples, replace=True)
        
    sampled_pts = norm_pts[choice]
    return torch.tensor(sampled_pts, dtype=torch.float32)

def convert_cloud2_to_numpy(msg):
    """Converts a sensor_msgs/msg/PointCloud2 message into an [N, 4] numpy array (x, y, z, intensity)."""
    # Simplified parser assuming standard KITTI-to-ROS2 point layout (x, y, z, intensity)
    point_step = msg.point_step
    data = msg.data
    num_points = len(data) // point_step
    
    points = np.frombuffer(data, dtype=np.float32, count=num_points * (point_step // 4))
    points = points.reshape((num_points, point_step // 4))[:, :4]
    return points

def convert_img_to_numpy(msg):
    """Converts a sensor_msgs/msg/Image message into an OpenCV BGR numpy array."""
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    
    if encoding == 'rgb8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == 'bgr8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
    else:
        # Fallback or handle mono8
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))

def visualize_all_sample(bin_path, gt_boxes=None, pred_boxes=None):
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
    if gt_boxes is not None:
        if isinstance(gt_boxes, np.ndarray) and gt_boxes.ndim == 1:
            gt_boxes = [gt_boxes]
        for gt_box in gt_boxes:
            geometries.append(create_o3d_box(gt_box, color=[0, 1, 0]))
    
    # Handle multiple prediction boxes
    if pred_boxes is not None and len(pred_boxes) > 0:
        print(pred_boxes)
        if isinstance(pred_boxes, np.ndarray) and pred_boxes.ndim == 1:
            pred_boxes = [pred_boxes]
        for pred_box in pred_boxes:
            geometries.append(create_o3d_box(pred_box, color=[1, 0, 0]))
        
    # 4. Open an interactive 3D visualization window (supports mouse rotation, translation, and zoom)
    print("=> Displaying 3D point cloud and bounding boxes (use mouse to rotate, zoom, and pan)")
    o3d.visualization.draw_geometries(geometries)
    
def view_model_results_from_bag():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=> Loading YOLO model from {yolo_path}...")
    yolo_model = YOLO(yolo_path)
    
    print(f"=> Loading ONNX model from {model_path}...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # Use ROS2_IRON to handle Jazzy-compatible message definitions
    typestore = get_typestore(Stores.ROS2_IRON)
    
    print(f"=> Opening ROS 2 bag: {bag_path}")
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        # Filter connections for image and point cloud topics
        connections = [c for c in reader.connections if c.topic in ['/kitti/point_cloud', '/kitti/image/gray/left']]
        
        # Simple frame synchronization queue or iteration logic
        # (Assuming sequential synchronized playback or matching timestamps)
        current_pc = None
        current_img = None
        frame_count = 0
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            if connection.topic == '/kitti/point_cloud':
                current_pc = convert_cloud2_to_numpy(msg)
            elif connection.topic == '/kitti/image/gray/left':
                current_img = convert_img_to_numpy(msg)
                
            # When both a frame and point cloud are available, process inference
            if current_pc is not None and current_img is not None:
                frame_count += 1
                print(f"\nProcessing ROS bag frame {frame_count}...")
                
                # 1. Run YOLO to get 2D boxes
                results = yolo_model(current_img, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy() # [xmin, ymin, xmax, ymax]
                
                gt_boxes_list = []
                pred_boxes_list = []
                
                for box2d in boxes:
                    # 2. Extract frustum points from live point cloud using helper function
                    frustum_pts, centroid, = extract_frustum_data_from_pc(current_pc, box2d)
                    if frustum_pts is None or len(frustum_pts) < 5:
                        continue
                    
                    # Normalize points relative to centroid (matching training preprocessing)
                    sampled_pts = sample_and_normalize_points(frustum_pts) # returns tensor [512, 3]
                
                    # 3. Model Inference
                    with torch.no_grad():
                        batch_points = sampled_pts.unsqueeze(0).numpy().astype(np.float32)
                        outputs = ort_session.run(None, {input_name: batch_points})
                        pred_box = outputs[0][0]
                        
                    # Map back to absolute camera coordinates
                    pred_box[0] += centroid[0]
                    pred_box[1] += centroid[1]
                    pred_box[2] += centroid[2]
                    
                    pred_boxes_list.append(pred_box)
            
                temp_bin_path = "/tmp/current_frame.bin"
                current_pc.astype(np.float32).tofile(temp_bin_path)
                
                visualize_all_sample(
                    bin_path=temp_bin_path,
                    gt_boxes=None,
                    pred_boxes=np.array(pred_boxes_list)
                )
            
                current_pc = None
                current_img = None
                
if __name__ == "__main__":
    view_model_results_from_bag()