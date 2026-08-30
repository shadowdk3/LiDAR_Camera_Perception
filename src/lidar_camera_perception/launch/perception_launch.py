import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the current package name
    package_name = 'lidar_camera_perception'

    camera_object_detection_node = Node(
        package=package_name, 
        executable='camera_object_detection.py', 
        name='camera_object_detection_node', 
        output='screen', 
        parameters=[], 
        remappings=[] 
    )

    # Configure the Lidar Preprocessing Node
    lidar_preprocessing_node = Node(
        package=package_name,
        executable='lidar_preprocessing',
        name='lidar_preprocessing_node',
        output='screen',
        parameters=[],
        remappings=[]
    )

    # Configure the Lidar Clustering/Detection Node
    lidar_cluster_object_node = Node(
        package=package_name,
        executable='lidar_cluster_object',
        name='lidar_cluster_object_node',
        output='screen',
        parameters=[],
        remappings=[]
    )

    # Configure the Lidar Tracker Node
    lidar_tracker_node = Node(
        package=package_name,
        executable='lidar_tracker',
        name='lidar_tracker_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    # Configure the Lidar Image Fusion Node
    lidar_image_fusion_node = Node(
        package=package_name,
        executable='lidar_image_fusion',
        name='lidar_image_fusion_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    # Create and return the LaunchDescription to execute both nodes simultaneously
    return LaunchDescription([
        camera_object_detection_node,
        lidar_preprocessing_node,
        lidar_cluster_object_node,
        lidar_tracker_node,
        lidar_image_fusion_node
    ])