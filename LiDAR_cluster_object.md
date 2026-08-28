# LiDAR cluster object
------------------------------------

- Cluster and mark each object

```
              ROS 2 
                │
                │ /lidar_preprocessing/object_pcd
                │ 
                ▼
            clustering
                │ 
                ▼
             objects
                │ 
                ▼
        3D bounding boxes
                │ 
                ▼
 ┌─────────────────────────────────────────┐
 │          ROS 2 Publishers               │
 │                                         │
 │  /lidar_cluster/clustered_obstacles_pcd │
 │  /lidar_cluster/bounding_boxes          │
 └─────────────────────────────────────────┘
```

- Receive input from object cloud, subscrib topic `/lidar_preprocessing/object_pcd`
- Define the cluster seaching method - Spatial Tree Search (Kd-Tree) to rapidly find neighboring points in 3D space without checking every single pair of points.
- Clustering
- Compute Axis-Aligned Bounding Box for clustering cloud

1. Create a node for LiDAR cluster object

```
cd lidar_camera_perception/src
code lidar_cluster_object.cpp
```

2. Create a launch file to launch all ROS2 node

```
cd lidar_camera_perception
mkdir launch
code perception_launch.py
```

3. Modify  `CMakeLists.txt` to compile all `.cpp`

- Add required package

```
find_package(visualization_msgs REQUIRED)
```

- CPP executable

```
add_executable(lidar_cluster_object src/lidar_cluster_object.cpp)
```

- Loop all cpp directories, definitions and libraries

```
set(ALL_CPP_TARGETS 
  lidar_preprocessing 
  lidar_cluster_object
)

foreach(target ${ALL_CPP_TARGETS})
  target_include_directories(${target} PUBLIC ${PCL_INCLUDE_DIRS})
  target_compile_definitions(${target} PUBLIC ${PCL_DEFINITIONS})
  target_link_libraries(${target} ${PCL_LIBRARIES})
endforeach()
```

- ament macro to link standard ROS2 dependencies

```
ament_target_dependencies(lidar_cluster_object
  rclcpp
  sensor_msgs
  pcl_conversions
  visualization_msgs                              
)
```

- Installion configuration

```
install(TARGETS ${ALL_CPP_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)
```

- install launch file

```
# launch
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
```

4. Build

```
colcon build --symlink-install --packages-select lidar_camera_perception
```

5. Test

- Run 

```
ros2 launch lidar_camera_perception perception_launch.py
```

- Run input

```
ros2 bag play kitti_dataset_0009
```

- rviz2 for visualization

```
rviz2
```

![lidar_cluster_object](./reference/lidar_cluster_object.gif)


