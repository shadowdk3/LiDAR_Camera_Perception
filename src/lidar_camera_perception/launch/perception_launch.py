import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Get the current package name
    package_name = 'lidar_camera_perception'

    # 1. Configure the Lidar Preprocessing Node
    lidar_preprocessing_node = Node(
        package=package_name,
        executable='lidar_preprocessing',
        name='lidar_preprocessing_node',
        output='screen',
        parameters=[
            # You can add parameter files or arguments here, for example:
            # {'min_x': -10.0, 'max_x': 10.0}
        ],
        remappings=[
            # Set topic remappings here if needed, for example:
            # ('/input_cloud', '/velodyne_points')
        ]
    )

    # 2. Configure the Lidar Clustering/Detection Node
    lidar_cluster_object_node = Node(
        package=package_name,
        executable='lidar_cluster_object',
        name='lidar_cluster_object_node',
        output='screen',
        parameters=[],
        remappings=[]
    )

    # 3. Configure the Lidar Tracker Node
    lidar_tracker_node = Node(
        package=package_name,
        executable='lidar_tracker',
        name='lidar_tracker_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    # Create and return the LaunchDescription to execute both nodes simultaneously
    return LaunchDescription([
        lidar_preprocessing_node,
        lidar_cluster_object_node,
        lidar_tracker_node
    ])