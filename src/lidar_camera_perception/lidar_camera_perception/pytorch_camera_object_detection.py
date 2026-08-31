#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from ros2_lidar_camera_perception.lidar_camera_perception.camera_object_detection_node import YoloNode
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T

# for the detection result, only consider the following classes: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
TARGET_CLASSES = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    6: 'bus',
    8: 'truck',
    10: 'traffic_light',
    12: 'stop_sign'
}

class PytorchCameraObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('camera_object_detection_node')

        # Detect if GPU (CUDA) is available, otherwise fallback to CPU
        self.device = torch.device('cuda' if torch.torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # Load the official pre-trained Faster R-CNN model weights (trained on COCO 91 classes)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation/inference mode

        # Define PyTorch transformation to normalize and convert images to Tensors
        self.transform = T.ToTensor()

        self.confidence_threshold = 0.5
        
        # subscribe to the camera image topic, and run the image_callback function when a new image is received
        self.sub_left_color = self.create_subscription(Image, '/kitti/image/color/left',self.image_callback, 10)

        self.pub_detection = self.create_publisher(Detection2DArray, '/camera/object_detections', 10)   # publishers for the detection results
        self.pub_annotated_frame = self.create_publisher(Image, '/camera/annotated_frame', 10)          # publishers for the annotated image
    
        self.get_logger().info("Camera Object Detection Node Started...")
        
    def image_callback(self, msg):
        # This function is called whenever a new image is received on the /kitti/image/color/left topic
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')   #convert the ROS Image message to an OpenCV image
            
            annotated_frame = cv_image.copy()           # Copy to annotated frame
                        
            # Pretrained PyTorch models expect RGB format, whereas OpenCV uses BGR
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Convert RGB image to a normalized PyTorch Tensor and move it to target device (GPU/CPU)
            input_tensor = self.transform(rgb_image).to(self.device)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]

            # Execute PyTorch model inference
            with torch.no_grad():
                predictions = self.model(input_batch)
                
            # Extract predictions for the first image in the batch and bring them back to CPU memory
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu()          # Coordinates formatted as [xmin, ymin, xmax, ymax]
            labels = prediction['labels'].cpu().tolist()
            scores = prediction['scores'].cpu().tolist()

            detection_array_msg = Detection2DArray()    # ROS 2 message container for multiple 2D object detections
            detection_array_msg.header = msg.header     # Set the header of the detection array to match the incoming image message's header (for timestamp and frame information)
            
            for box, label, score in zip(boxes, labels, scores):
                # Only process the 8 specified traffic classes above your confidence threshold
                if label in TARGET_CLASSES and score >= self.confidence_threshold:
                
                    class_name = TARGET_CLASSES[label]
        
                    # Unpack the absolute corner coordinates [xmin, ymin, xmax, ymax] from PyTorch output
                    xmin, ymin, xmax, ymax = box.tolist()
                
                    # FORMAT CONVERSON: Convert [xmin, ymin, xmax, ymax] to ROS vision_msgs [cx, cy, width, height] format
                    width = xmax - xmin
                    height = ymax - ymin
                    x_center = xmin + (width / 2.0)
                    y_center = ymin + (height / 2.0)
        
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
                    # hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)
            
                    detection_array_msg.detections.append(detection)
                
                    # draw annotated frame
                    # Format float metrics to pixel integers for OpenCV renderer execution
                    ixmin, iymin, ixmax, iymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    label_text = f"{class_name}: {score:.2f}"
                                
                    # Draw a solid red backdrop box wrapper block directly behind text layout
                    cv2.rectangle(annotated_frame, (ixmin, iymin), (ixmax, iymax), (0, 0, 255), 2)
                
                    # Estimate bounding space envelope constraints for background label flag tag
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Stamp anti-aliased bright white indicator text across the red background block
                    cv2.putText(annotated_frame, label_text, (ixmin, iymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.pub_detection.publish(detection_array_msg)                     # Publish the detection results to the /camera/object_detections topic

            # pulbish annotated frame
            annotated_frame_msg = CvBridge().cv2_to_imgmsg(annotated_frame, encoding="passthrough")   # Convert the annotated OpenCV image back to a ROS Image message
            annotated_frame_msg.header = msg.header
            self.pub_annotated_frame.publish(annotated_frame_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

def main(args=None):
    rclpy.init(args=args)                       # init ROS 2 Python client library
    node = PytorchCameraObjectDetectionNode()   # Instantiate your custom node class 
    rclpy.spin(node)                            # This keeps the node alive to listen for events. It constantly processes incoming data
    node.destroy_node()                         # When the user presses Ctrl+C, rclpy.spin() unblocks. This line cleanups the node 
    rclpy.shutdown()                            # This cleanly disconnects the application from the ROS 2 graph and releases all system resources
    
if __name__ == '__main__':
    main()