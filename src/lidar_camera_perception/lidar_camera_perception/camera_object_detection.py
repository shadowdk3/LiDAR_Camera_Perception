#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from ros2_lidar_camera_perception.lidar_camera_perception.camera_object_detection_node import YoloNode
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

from db_logger import DatabaseLogger  # Use relative or absolute package import based on folder layout

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

class CameraObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('camera_object_detection_node')

        # YOLO setting
        self.model = YOLO("models/yolo11n.pt")
        self.confidence_threshold = 0.5
        
        # Initialize Database Logger instance
        self.db = DatabaseLogger(self.get_logger())
        
        # subscribe to the camera image topic, and run the image_callback function when a new image is received
        self.sub_left_color = self.create_subscription(Image, '/kitti/image/color/left',self.image_callback, 10)

        self.pub_detection = self.create_publisher(Detection2DArray, '/camera/object_detections', 10)   # publishers for the detection results
        self.pub_annotated_frame = self.create_publisher(Image, '/camera/yolo_detections', 10)          # publishers for the annotated image
    
        self.get_logger().info("Camera Object Detection Node Started...")
        
    def image_callback(self, msg):
        # This function is called whenever a new image is received on the /kitti/image/color/left topic
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')   #convert the ROS Image message to an OpenCV image
            results = self.model(
                cv_image, 
                conf=self.confidence_threshold, 
                classes=list(TARGET_CLASSES.values()), 
                verbose=False
            )
            
            result = results[0]
                
            detection_array_msg = Detection2DArray()    # ROS 2 message container for multiple 2D object detections
            detection_array_msg.header = msg.header     # Set the header of the detection array to match the incoming image message's header (for timestamp and frame information)
            
            # Extract header parameters outside the bounding box loops
            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec
            frame_id_str = msg.header.frame_id
            
            # do every result.boxes in the detection result
            for box in result.boxes:
                x_center, y_center, width, height = box.xywh[0].tolist()        # get box size
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.model.names[class_id]
                
                # print the detection result to the console
                # self.get_logger().info(
                #     f"[YOLO detected] no.: {len(result.boxes)} | "
                #     f"class_name: {class_name} | confidence: {confidence}"
                # )
                self.db.log_detection(
                    sec, nanosec, frame_id_str, class_name, confidence, 
                    x_center, y_center, width, height
                )
                
                # create a Detection2D message for each detected object
                detection = Detection2D()
                detection.header = msg.header
                
                # define the bounding box for the detected object
                detection.bbox.center.position.x = x_center
                detection.bbox.center.position.y = y_center
                detection.bbox.size_x = width
                detection.bbox.size_y = height
                            
                # 3D space pose message object
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence
                detection.results.append(hypothesis)
           
                detection_array_msg.detections.append(detection)
            
            # Frame ended processing, execute a batch database disk write commit
            self.db.commit_frame()
            
            self.pub_detection.publish(detection_array_msg)                     # Publish the detection results to the /camera/object_detections topic
            
            annotated_frame = result.plot()                                                     # Draw the bounding boxes and labels on the original image for visualization
            annotated_frame_msg = CvBridge().cv2_to_imgmsg(annotated_frame, encoding="passthrough")   # Convert the annotated OpenCV image back to a ROS Image message
            annotated_frame_msg.header = msg.header
            self.pub_annotated_frame.publish(annotated_frame_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def destroy_node(self):
        # Disconnect from database gracefully on termination sequences
        self.db.close()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)               # init ROS 2 Python client library
    node = CameraObjectDetectionNode()  # Instantiate your custom node class 
    
    try:
        rclpy.spin(node)                    # This keeps the node alive to listen for events. It constantly processes incoming data
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()                 # When the user presses Ctrl+C, rclpy.spin() unblocks. This line cleanups the node 
        rclpy.shutdown()                    # This cleanly disconnects the application from the ROS 2 graph and releases all system resources
    
if __name__ == '__main__':
    main()