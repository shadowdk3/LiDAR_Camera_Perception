import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from torch.utils.tensorboard import SummaryWriter
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
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, points):
        x = points.permute(0, 2, 1)
        x = self.mlp1(x)
        x = torch.max(x, dim=2)[0]
        return self.fc(x)

def read_kitti_calib():
    velo_to_cam0_extrinsic = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
        [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    cam0_rectification = np.array([
        [9.999239e-01, 9.837760e-03, -7.445048e-03, 0.000000e+00],
        [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.000000e+00],
        [7.402527e-03, 4.351614e-03, 9.999631e-01, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]
    ])
    cam2_projection_rectified = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])
    return cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_extrinsic, velo_to_cam0_extrinsic

def parse_kitti_tracklets(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracklets = []
    for item in root.findall('.//item'):
        obj_type = item.find('objectType').text if item.find('objectType') is not None else 'Unknown'
        tracklet = {
            'objectType': obj_type,
            'h': float(item.find('h').text) if item.find('h') is not None else 0.0,
            'w': float(item.find('w').text) if item.find('w') is not None else 0.0,
            'l': float(item.find('l').text) if item.find('l') is not None else 0.0,
            'first_frame': int(item.find('first_frame').text) if item.find('first_frame') is not None else 0,
            'poses': []
        }
        poses_node = item.find('poses')
        if poses_node is not None:
            for pose_item in poses_node.findall('item'):
                tracklet['poses'].append({
                    'tx': float(pose_item.find('tx').text),
                    'ty': float(pose_item.find('ty').text),
                    'tz': float(pose_item.find('tz').text),
                    'rz': float(pose_item.find('rz').text)
                })
        tracklets.append(tracklet)
    return tracklets

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = "/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync"
    img_path = os.path.join(data_dir, "image_00/data/0000000072.png")
    bin_path = os.path.join(data_dir, "velodyne_points/data/0000000072.bin")
    label_path = os.path.join(data_dir, "tracklet_labels.xml")
    
    img = cv2.imread(img_path)
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    annotations = parse_kitti_tracklets(label_path)
    proj, ext = read_kitti_calib()
    
    # Load trained model weights
    model = SimpleFrustumPointNet().to(device)
    model.load_state_dict(torch.load('frustum_pointnet.pth'))
    # model.load_state_dict(torch.load('frustum_pointnet_30epoch.pth'))
    model.eval()
    
    # Initialize TensorBoard writer for evaluation/inference logging
    writer = SummaryWriter(log_dir='runs/frustum_pointnet_eval')
    
    yolo = YOLO("/home/user/LiDAR_Camera_Perception_ws/yolo11n.pt")
    results = yolo(img, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    
    current_frame_idx = int(os.path.basename(img_path).split('.')[0])
    
    total_eval_loss = 0.0
    valid_detections_count = 0
    
    # 1. Run inference and draw Predicted 3D Boxes (Red)
    with torch.no_grad():
        for box2d, cls_id in zip(boxes, clss):
            if int(cls_id) in TARGET_CLASSES.values():
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
                
                if len(frustum_pts) > 5:
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
                    pred_box[:3] += centroid
                    
                    # Draw Prediction in Red (B, G, R) -> (0, 0, 255)
                    img = draw_3d_box(img, pred_box, proj, (0, 0, 255))

                    # Compute evaluation loss against matched ground truth if available
                    for tracklet in annotations:
                        pose_idx = current_frame_idx - tracklet['first_frame']
                        if 0 <= pose_idx < len(tracklet['poses']):
                            p = tracklet['poses'][pose_idx]
                            gt_box = np.array([p['tx'], p['ty'], p['tz'], tracklet['l'], tracklet['w'], tracklet['h'], p['rz']], dtype=np.float32)
                            gt_box[:3] -= centroid
                            
                            pred_tensor = torch.tensor(pred_box, dtype=torch.float32).unsqueeze(0)
                            gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
                            loss = F.smooth_l1_loss(pred_tensor, gt_tensor).item()
                            
                            total_eval_loss += loss
                            valid_detections_count += 1
                            break
                 
    # 2. Draw Ground Truth 3D Boxes (Green) for comparison
        for tracklet in annotations:
            pose_idx = current_frame_idx - tracklet['first_frame']
            if 0 <= pose_idx < len(tracklet['poses']):
                p = tracklet['poses'][pose_idx]
                gt_box = np.array([p['tx'], p['ty'], p['tz'], tracklet['l'], tracklet['w'], tracklet['h'], p['rz']], dtype=np.float32)
                # Draw GT in Green -> (0, 255, 0)
                img = draw_3d_box(img, gt_box, proj, (0, 255, 0))
                       
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