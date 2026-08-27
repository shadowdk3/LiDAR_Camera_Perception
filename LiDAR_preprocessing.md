# LiDAR preprocessing
------------------------------------

- Seperate ground and object in LiDAR data

- Core Flow

```
              ROS 2 
                │
                │ /kitti/point_cloud
                │ 
                ▼
 ┌─────────────────────────────────────────┐
 │ sensor_msgs/msg/PointCloud2             │
 │              │                          │
 │              ▼ (pcl_conversions)        │
 │       PCL PointCloud                    │
 │              │                          │
 │              ▼ (pcl::VoxelGrid)         │
 │       VoxelGrid Downsample              │
 │              │                          │
 │              ▼ (pcl::SACSegmentation)   │
 │   RANSAC Plane Segmentation             │
 │              │                          │
 │      ┌───────┴───────┐                  │
 │      ▼               ▼                  │
 │  [Success]       [Failure] ───┐         │
 │      │                        │         │
 │      ▼ (pcl::ExtractIndices)  │         │
 │  ┌───────────────┐            │         │
 │  │ Ground Points │            │         │
 │  └───────┬───────┘            │         │
 │          │                    │         │
 │          ▼                    │         │
 │  ┌───────────────┐            │         │
 │  │ Obstacles     │            │         │
 │  └───────┬───────┘            │         │
 └──────────┼────────────────────┼─────────┘
            │                    │
            ├────────────────────┘ (Drop / Log Error)
            ▼
 ┌─────────────────────────────────────────┐
 │          ROS 2 Publishers               │
 │                                         │
 │  /lidar_preprocessing/ground_pcd        │
 │  /lidar_preprocessing/object_pcd        │
 └─────────────────────────────────────────┘
```

- Receive input from LiDAR, subscrib topic `/kitti/point_cloud`
- Covert ROS2 PCL to PCL for sepearte ground and object
- Downsample voxel
- Use RANSAC plane segmentation method to seperate ground and object
- Publish ground PCL and object PCL

1. Create a node for LiDAR preprocessing

```
cd lidar_camera_perception/src
code lidar_preprocessing.cpp
```

2. Modify  `CMakeLists.txt` to compile LiDAR_preprocessing

- Add required package

```
find_package(PCL REQUIRED COMPONENTS common filters io segmentation sample_consensus) # PLC
find_package(pcl_conversions REQUIRED)
```

- CPP executable

```
add_executable(lidar_preprocessing src/lidar_preprocessing.cpp)
```

- PLC directory, definitions, library for PCL

```
# Include PCL directories
target_include_directories(lidar_preprocessing PUBLIC
  ${PCL_INCLUDE_DIRS}
)

# Compile definitions for PCL
target_compile_definitions(lidar_preprocessing PUBLIC
  ${PCL_DEFINITIONS}
)

# Link ROS2 and PCL libraries
target_link_libraries(lidar_preprocessing ${PCL_LIBRARIES})
```

- ament macro to link standard ROS2 dependencies

```
ament_target_dependencies(lidar_preprocessing
  rclcpp                        # for ROS2 C++ client library
  sensor_msgs                   # for PointCloud2 messages
  pcl_conversions               # for converting between PCL and ROS2 PointCloud2 messages
)
```

- Installion configuration

```
install(TARGETS lidar_preprocessing
  DESTINATION lib/${PROJECT_NAME}
)
```

3. Build

```
colcon build --symlink-install --packages-select lidar_camera_perception
```

5. Test

- Run object detection

```
ros2 run lidar_camera_perception lidar_preprocessing
```

- Run input

```
ros2 bag play kitti_dataset_0009
```

- rviz2 for visualization

```
rviz2
```

![lidar_preprocessing](./reference/lidar_preprocessing.gif)