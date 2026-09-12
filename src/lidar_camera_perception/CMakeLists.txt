cmake_minimum_required(VERSION 3.8)
project(lidar_camera_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(vision_msgs REQUIRED)  #  Detection2DArray
find_package(sensor_msgs REQUIRED)  #  Image
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io segmentation sample_consensus) # PLC
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

include_directories(${ONNXRUNTIME_ROOT}/include)

find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime 
    PATHS ${ONNXRUNTIME_ROOT}/lib 
    REQUIRED
)

# cpp executables
add_executable(lidar_preprocessing src/lidar_preprocessing.cpp)
add_executable(lidar_cluster_object src/lidar_cluster_object.cpp)
add_executable(lidar_tracker src/lidar_tracker.cpp)
add_executable(lidar_image_fusion src/lidar_image_fusion.cpp)
add_executable(lidar_frustum_onnx src/lidar_frustum_onnx.cpp)

set(ALL_CPP_TARGETS 
  lidar_preprocessing 
  lidar_cluster_object
  lidar_tracker
  lidar_image_fusion
  lidar_frustum_onnx
)

foreach(target ${ALL_CPP_TARGETS})
  target_include_directories(${target} PUBLIC ${PCL_INCLUDE_DIRS})
  target_compile_definitions(${target} PUBLIC ${PCL_DEFINITIONS})
  target_link_libraries(${target} ${PCL_LIBRARIES})
endforeach()

target_include_directories(lidar_frustum_onnx PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include
)

target_link_libraries(lidar_frustum_onnx
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
)

# ament macro to link standard ROS2 dependencies (rclcpp, sensor_msgs, etc.)
# for cpp
ament_target_dependencies(lidar_preprocessing
  rclcpp                        # for ROS2 C++ client library
  sensor_msgs                   # for PointCloud2 messages
  pcl_conversions               # for converting between PCL and ROS2 PointCloud2 messages
)

ament_target_dependencies(lidar_cluster_object
  rclcpp
  sensor_msgs
  pcl_conversions
  visualization_msgs                              
)

ament_target_dependencies(lidar_tracker
  rclcpp
  sensor_msgs
  visualization_msgs                              
)

ament_target_dependencies(lidar_image_fusion
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  cv_bridge                           
)

ament_target_dependencies(lidar_frustum_onnx
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  cv_bridge         
  message_filters    
)

# --- INSTALLATION CONFIGURATION ---
# python scripts
install(PROGRAMS
  lidar_camera_perception/camera_object_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  lidar_camera_perception/pytorch_camera_object_detection.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  lidar_camera_perception/frustum_onnx_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# C++ executable
install(TARGETS ${ALL_CPP_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

# launch
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
