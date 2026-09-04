"""
KITTI 3D Tracklet Cleaner and Visualizer

This script processes KITTI dataset tracklet XML files by:
    1. Parsing 3D bounding box dimensions and trajectory poses.
    2. Transforming box vertices into the camera coordinate system using calibration matrices.
    3. Filtering out abnormal or overly close poses where depth (Z <= 0.1).
    4. Automatically removing invalid pose entries and saving the cleaned XML file.
    5. Projecting and rendering valid 3D bounding boxes onto synchronized camera images.
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil

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

def load_kitti_calib():
    """Load KITTI LiDAR-to-camera extrinsic and projection calibration matrices."""
    velo_to_cam0_ext = np.array([
        [ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])
    cam0_rectification = np.array([
        [ 9.999239e-01,  9.837760e-03, -7.445048e-03,  0.000000e+00],
        [-9.869795e-03,  9.999421e-01, -4.278459e-03,  0.000000e+00],
        [ 7.402527e-03,  4.351614e-03,  9.999631e-01,  0.000000e+00],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])
    cam2_projection_rectified = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])
    return cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_ext, velo_to_cam0_ext
    
def parse_tracklets(xml_path, velo_to_cam0_ext, output_xml_path):
    """Parse XML tracklet file, filter out poses with invalid depth, and save cleaned XML."""
    
    shutil.copyfile(xml_path, output_xml_path)
    
    tree = ET.parse(output_xml_path)
    root = tree.getroot()
    tracklets = []
    
    items = root.findall(".//item")
    print(f"number of items: {len(items)}")
    
    for item in items:
        obj_type_node = item.find('objectType')
        obj_type = obj_type_node.text if obj_type_node is not None else 'Unknown'
        
        h = float(item.find('h').text) if item.find('h') is not None else 0.0
        w = float(item.find('w').text) if item.find('w') is not None else 0.0
        l = float(item.find('l').text) if item.find('l') is not None else 0.0
        ff_node = int(item.find('first_frame').text) if item.find('first_frame') is not None else 0
        
        tracklet = {
            'objectType': obj_type,
            'h': h,
            'w': w,
            'l': l,
            'first_frame': ff_node,
            'poses': []
        }
        
        poses_node = item.find('poses')
        if poses_node is not None:
            for pose_item in poses_node.findall('item'):
                tx = float(pose_item.find('tx').text) if pose_item.find('tx') is not None else 0.0
                ty = float(pose_item.find('ty').text) if pose_item.find('ty') is not None else 0.0
                tz = float(pose_item.find('tz').text) if pose_item.find('tz') is not None else 0.0
                rz = float(pose_item.find('rz').text) if pose_item.find('rz') is not None else 0.0
                
                # Construct 3D bounding box corners
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [0, 0, 0, 0, h, h, h, h]
                corners = np.vstack([x_corners, y_corners, z_corners])
                            
                # Apply rotation and translation
                c, s = np.cos(rz), np.sin(rz)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                corners = R @ corners
                corners[0, :] += tx
                corners[1, :] += ty
                corners[2, :] += tz
                            
                # Convert to homogeneous coordinates and project to camera system for depth validation
                ones = np.ones((1, corners.shape[1]))
                corners_homo = np.vstack((corners, ones))
                corners_cam = velo_to_cam0_ext @ corners_homo
            
                # Safety check: skip and remove pose if any vertex depth Z <= 0.1 (behind or too close to camera)
                if np.any(corners_cam[2, :] < 0.1):
                    # print(ET.tostring(item, encoding='utf-8').decode('utf-8'))  # print item
                    poses_node.remove(pose_item)    # remove item (inner item)
                    continue
                            
                tracklet['poses'].append({
                    'tx': tx,
                    'ty': ty,
                    'tz': tz,
                    'rz': rz
                })
                
        tracklets.append(tracklet)
        
    # Write updated XML back to disk
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    
    return tracklets

def run_visualization(base_dir):
    """Load images and visualize valid 3D bounding boxes projected onto camera frames."""
    image_dir = os.path.join(base_dir, "image_00", "data")
    xml_path = os.path.join(base_dir, "tracklet_labels.xml")
    output_xml_path = os.path.join(base_dir, "tracklet_labels_cleaned.xml")
    
    projection_matrix, velo_to_cam0_ext = load_kitti_calib()
    tracklets = parse_tracklets(xml_path, velo_to_cam0_ext, output_xml_path)
    
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    print(f"Number of tracklets: {len(tracklets)}")
    print(f"Number of images: {len(images)}")
    
    for img_idx, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        
        for tracklet in tracklets:
            start_frame = tracklet['first_frame']
            pose_idx = img_idx - start_frame
            
            if 0 <= pose_idx < len(tracklet['poses']):
                pose = tracklet['poses'][pose_idx]
                h, w, l = tracklet['h'], tracklet['w'], tracklet['l']
                tx, ty, tz, rz = pose['tx'], pose['ty'], pose['tz'], pose['rz']
                
                # Construct 3D box corners
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [0, 0, 0, 0, h, h, h, h]
                corners = np.vstack([x_corners, y_corners, z_corners])
                
                # Apply rotation and translation
                c, s = np.cos(rz), np.sin(rz)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                corners = R @ corners
                corners[0, :] += tx
                corners[1, :] += ty
                corners[2, :] += tz
                
                # Project 3D points to 2D image plane
                ones = np.ones((1, corners.shape[1]))
                corners_homo = np.vstack((corners, ones))
                img_coords = projection_matrix @ corners_homo
                x = img_coords[0, :] / img_coords[2, :]
                y = img_coords[1, :] / img_coords[2, :]
                pts_2d = np.vstack((x, y)).T.astype(int)
                
                # Draw 12 edges of the 3D bounding box
                lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                for start, end in lines:
                    cv2.line(img, tuple(pts_2d[start]), tuple(pts_2d[end]), (0, 255, 0), 2)
                    
        cv2.imshow("KITTI Tracklets Python Viewer", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

run_visualization("/home/user/LiDAR_Camera_Perception_ws/data/2011_09_26/2011_09_26_drive_0009_sync")