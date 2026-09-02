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

1. **2D Detection:** YOLOv8 detects objects in synchronized RGB images and extracts 2D bounding boxes $(u_1, v_1, u_2, v_2)$.
2. **Calibration & Projection:** Loads raw KITTI calibration files (`calib_velo_to_cam.txt`, `calib_cam_to_cam.txt`) and pre-computes the unified 3D-to-2D projection matrix:
   $$\mathbf{M}_{\text{proj}} = \mathbf{P}_{\text{rect\_02}} \times \mathbf{R}_{\text{rect\_00}} \times \mathbf{T}_{\text{velo\_to\_cam0}}$$
3. **Frustum Cropping:** Projects LiDAR points onto the image plane, filters out points behind the camera ($Z \le 0$), and isolates 3D points falling within each YOLO 2D bounding box.
4. **3D Box Estimation:** Feeds the cropped 3D frustum point cloud into Frustum PointNet to regress oriented 3D bounding boxes.


---------------------------

## Issue

- cannot view 3d in wsl

```
export XDG_SESSION_TYPE=x11
unset WAYLAND_DISPLAY
```