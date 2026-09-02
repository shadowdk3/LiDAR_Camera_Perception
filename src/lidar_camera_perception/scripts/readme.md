# Frustum-to-Voxel 3D Object Detection for Autonomous Systems
-------------------------------------

## Install CUDA

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-libraries-dev-12-1
```

- Configure Environment Paths

```
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

- Verify Compiler Installation

```
nvcc --version
```

## Pytorch

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- because of this pytorch will upgrade numpy, so need to uninstall numpy and install numpy < 2.0

```
pip install "tifffile<2025.0.0"
pip install "numpy<2.0"
```

## Install Dependencies

```
pip install open3d
pip install "scipy<1.14"
sudo apt install libgl1 libegl1 libxrandr2 libxinerama1 libxcursor1 libxi6 mesa-utils
sudo apt update && sudo apt install -y libgles2 libgles2-mesa-dev
```

## Repository Structure

```
script/
├── test_frustum_crop.py         # Main test script for projection, frustum extraction, and Open3D visualization
└── README.md
```

## LiDAR-Camera Fusion & Frustum PointNet 3D Detection

**Significantly reducing 3D spatial search and computational burden:**
Using **2D YOLO bounding boxes** to crop **raw LiDAR point clouds** into localized frustums dramatically **reduces computational costs**, allowing 3D models to focus exclusively on relevant object regions instead of processing the entire scene.

A modular perception pipeline that combines 2D object detection (YOLO) with LiDAR point clouds using KITTI calibration parameters to extract 3D frustums and predict accurate 3D bounding boxes.

### Pipeline Architecture

- test_frustum_crop.py

1. **2D Detection:** YOLOv8 detects objects in synchronized RGB images and extracts 2D bounding boxes $(u_1, v_1, u_2, v_2)$.
2. **Calibration & Projection:** Loads raw KITTI calibration files (`calib_velo_to_cam.txt`, `calib_cam_to_cam.txt`) and pre-computes the unified 3D-to-2D projection matrix:
   $$\mathbf{M}_{\text{proj}} = \mathbf{P}_{\text{rect\_02}} \times \mathbf{R}_{\text{rect\_00}} \times \mathbf{T}_{\text{velo\_to\_cam0}}$$
3. **Frustum Cropping:** Projects LiDAR points onto the image plane, filters out points behind the camera ($Z \le 0$), and isolates 3D points falling within each YOLO 2D bounding box.
4. **3D Box Estimation:** Feeds the cropped 3D frustum point cloud into Frustum PointNet to regress oriented 3D bounding boxes.

- frustum_pointnet_pipeline.py

1. **Core Purpose:** End-to-end prototype validating a Frustum-to-Voxel 3D object detection framework.

2. **2D-3D Cross-Domain Integration:** Leverages YOLOv11 for 2D bounding box detection and applies KITTI calibration matrices to project and crop corresponding LiDAR 3D point cloud frustums.

3. **Data Normalization & Batch Alignment:** Applies local zero-centering to cropped point clouds, standardizing dynamic point counts into a fixed tensor shape of (Batch, 512, 3) via random sampling and zero-padding.

4. **PyTorch Tensor Flow Validation:** Tests the custom SimpleFrustumPointNet model to ensure proper tensor ingestion, feature extraction, and regression head output for 7D 3D bounding box parameters.

   - Center Coordinates ($x, y, z$): The 3D center position of the object relative to the sensor frame in meters (e.g., $x = 12.5$, $y = 1.2$, $z = -0.5$).
   - Dimensions ($l, w, h$): The physical size of the object in meters, representing length, width, and height (e.g., length $= 4.2$, width $= 1.8$, height $= 1.5$ for a sedan).
   - Orientation ($\theta$): The yaw rotation angle around the vertical axis in radians (e.g., $\theta = 0.15$).

   ```
   # [x, y, z, length, width, height, yaw]
   bounding_box_7d = [12.5, 1.2, -0.5, 4.2, 1.8, 1.5, 0.15]
   ```

5. **Loss Calculation & Optimization Validation:** Computes Smooth L1 Loss (F.smooth_l1_loss) between the model's 7D regression outputs and aligned ground truth targets to evaluate prediction errors and validate gradient-ready optimization loops.

---------------------------

## Issue

- cannot view 3d in wsl

```
export XDG_SESSION_TYPE=x11
unset WAYLAND_DISPLAY
```