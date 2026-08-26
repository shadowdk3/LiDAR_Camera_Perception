# Generate_ROS2_BAG
----------------------------------

- Use [ros2_kitti_publishers](https://github.com/umtclskn/ros2_kitti_publishers) tool to play raw dataseet as ROS2 topic

## Directory Structure

if the folder name is different, need to change in source code

```
ros2_ws
|- src
|- install
|- log
|- build
|- data
|-----|- 2011_09_26
|-----|----------|- 2011_09_26_drive_0009_sync
|--------------------------------------------|- image_00
|--------------------------------------------|- image_01
|--------------------------------------------|- image_02
|--------------------------------------------|- image_03
|--------------------------------------------|- oxts
|--------------------------------------------|- velodyne_points
```

## Build and Run 

- Build package

```
colcon build --symlink-install
```

- Open one terminal run the ROS2 node

```
ros2 run ros2_kitti_publishers kitti_publishers
```

- Open other terminal 

```
ros2 topic list
```

should see

```
/clicked_point
/goal_pose
/initialpose
/kitti/image/color/left
/kitti/image/color/right
/kitti/image/gray/left
/kitti/image/gray/right
/kitti/imu
/kitti/marker_array
/kitti/nav_sat_fix
/kitti/point_cloud
/parameter_events
/rosout
/tf
/tf_static
```

- Then run rviz for visualization 

```
rviz2
```
- And add topics into display

- Record ROS2 bag, open new terminal, should run the command before play the node

![rviz2](./reference/kitti_rosbag.gif)

```
ros2 bag record -a -o kitti_dataset_0009
```

- Play the node again

```
ros2 run ros2_kitti_publishers kitti_publishers
```

- After recording, press Ctrl + C

- Check info for the bag

```
ros2 bag info kitti_dataset_0009
```

- Playback 

```
ros2 bag play kitti_dataset_0009
```
