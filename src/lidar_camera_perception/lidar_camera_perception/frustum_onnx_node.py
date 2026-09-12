#!/usr/bin/env python3

import os 

import rclpy
import onnxruntime as ort

from rclpy.node import Node
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import message_filters
import sensor_msgs_py.point_cloud2 as pc2

SCRIPT_DIR = Path(__file__).resolve().parent.parent
WS_DIR = SCRIPT_DIR.parent.parent
model_path = os.path.join(SCRIPT_DIR, "frustum_pointnet", "checkpoints", "frustum_pointnet_fine_tune.onnx")
yolo_path = os.path.join(WS_DIR, "models", "yolo11n.pt")

class FrustumOnnxNode(Node):
    def __init__(self):
        super().__init__('frustum_onnx_node')
        
        self.get_logger().info(f"Loading YOLO model from {yolo_path}...")
        self.yolo_model = YOLO(yolo_path)
        
        self.get_logger().info(f"Loading ONNX model from {model_path}...")
        self.ort_session = ort.InferenceSession(
            model_path, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # ROS 2 Subscribers & Publishers
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, '/kitti/point_cloud')
        self.img_sub = message_filters.Subscriber(self, Image, '/kitti/image/gray/left')
        
        self.marker_pub = self.create_publisher(MarkerArray, '/frustum_bounding_boxes', 10)
        self.debug_pc_pub = self.create_publisher(PointCloud2, '/frustum_debug_point_cloud', 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, self.img_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        self.frame_count = 0
        self.get_logger().info("Frustum PointNet Node initialized and listening to topics...")
    
    def read_kitti_calib(self):
        velo_to_cam0_extrinsic = np.array([
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

        velo_to_cam2_projection = cam2_projection_rectified @ cam0_rectification @ velo_to_cam0_extrinsic
        return velo_to_cam2_projection, velo_to_cam0_extrinsic, None

    def extract_frustum_data_from_pc(self, point_cloud, box2d):
        velo_cam2_projection, velo_to_cam0_extrinsic, _ = self.read_kitti_calib()
        
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

    def sample_and_normalize_points(self, frustum_pts, num_samples=512):
        centroid = np.mean(frustum_pts, axis=0)
        norm_pts = frustum_pts - centroid
        
        num_pts = len(norm_pts)
        if num_pts >= num_samples:
            choice = np.random.choice(num_pts, num_samples, replace=False)
        else:
            choice = np.random.choice(num_pts, num_samples, replace=True)
            
        sampled_pts = norm_pts[choice]
        return torch.tensor(sampled_pts, dtype=torch.float32)

    def convert_cloud2_to_numpy(self, msg):
        point_step = msg.point_step
        data = msg.data
        num_points = len(data) // point_step
        points = np.frombuffer(data, dtype=np.float32, count=num_points * (point_step // 4))
        return points.reshape((num_points, point_step // 4))[:, :4]

    def convert_img_to_numpy(self, msg):
        height = msg.height
        width = msg.width
        encoding = msg.encoding
        
        if encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == 'bgr8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        else:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))
        
    def yaw_to_quaternion(self, yaw):
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw * 0.5)
        qw = np.cos(yaw * 0.5)
        return qx, qy, qz, qw
    
    def publish_marker_array(self, pred_boxes, header):
        marker_array = MarkerArray()

        del_marker = Marker()
        del_marker.header = header
        del_marker.action = Marker.DELETEALL
        marker_array.markers.append(del_marker)

        if pred_boxes is not None:
            for i, box in enumerate(pred_boxes):
                x, y, z, l, w, h, rz = box

                marker = Marker()
                marker.header = header
                marker.ns = "frustum_3d_boxes"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = float(x)
                marker.pose.position.y = float(y)
                marker.pose.position.z = float(z)

                qx, qy, qz, qw = self.yaw_to_quaternion(rz)
                marker.pose.orientation.x = qx
                marker.pose.orientation.y = qy
                marker.pose.orientation.z = qz
                marker.pose.orientation.w = qw

                marker.scale.x = float(l)
                marker.scale.y = float(w)
                marker.scale.z = float(h)

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.6

                marker.lifetime.sec = 0
                marker.lifetime.nanosec = int(2e8)

                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        
    def publish_debug_point_cloud(self, points_np, header):
        # points_np is expected to be an (N, 3) or (N, 4) numpy array
        debug_msg = pc2.create_cloud_xyz32(header, points_np[:, :3])
        self.debug_pc_pub.publish(debug_msg)
    
    def sync_callback(self, pc_msg, img_msg):
        self.frame_count += 1
        self.get_logger().info(f"Processing synchronized frame {self.frame_count}...")
        
        current_pc = self.convert_cloud2_to_numpy(pc_msg)
        current_img = self.convert_img_to_numpy(img_msg)
        
        results = self.yolo_model(current_img, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        pred_boxes_list = []
        all_frustum_pts = []
        
        for box2d in boxes:
            frustum_pts, centroid = self.extract_frustum_data_from_pc(current_pc, box2d)
            if frustum_pts is None or len(frustum_pts) < 5:
                continue

            all_frustum_pts.append(frustum_pts)
            sampled_pts = self.sample_and_normalize_points(frustum_pts)

            with torch.no_grad():
                batch_points = sampled_pts.unsqueeze(0).numpy().astype(np.float32)
                outputs = self.ort_session.run(None, {self.input_name: batch_points})
                pred_box = outputs[0][0]

            pred_box[0] += centroid[0]
            pred_box[1] += centroid[1]
            pred_box[2] += centroid[2]
            pred_boxes_list.append(pred_box)

        pred_array = np.array(pred_boxes_list) if pred_boxes_list else None
        self.publish_marker_array(pred_array, pc_msg.header)
        
        # Publish combined frustum points to visually verify what the network "saw"
        if len(all_frustum_pts) > 0:
            combined_pts = np.vstack(all_frustum_pts)
            self.publish_debug_point_cloud(combined_pts, pc_msg.header)
            
        
if __name__ == '__main__':
    rclpy.init()
    node = FrustumOnnxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()