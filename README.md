# 3d-object-detection-ros-
This repository implements 3D object detection for vehicles using PointPillars AI algorithm, based on ROS and Docker. It processes 16-channel Velodyne point cloud data as input for real-time 3D vehicle detection.

# Overview
This project uses a 16-channel Velodyne LiDAR (VLP-16) to acquire point cloud data and performs PointPillars-based 3D object detection in a ROS and Docker environment.

# Demo
<img width="856" height="650" alt="image" src="https://github.com/user-attachments/assets/182a40f3-5eb2-48c6-a05e-484b4e11990f" />


# Environment & Requirements
### Hardware
- GPU: NVIDIA RTX 4060 (Laptop)
- Sensor: Veloyne VLP-16(16-chaanel)

### Host(Laptop)
- OS: Ubuntu 22.04 (running on external SSD)
- NVIDIA Driver: 580.95.05
- Docker: used to isolate ROS/AI environments
- Networking: containers communicate via ROS master (host or one container) / host networking recommended

### Software Versions (per container)
This project separates the sensor driver and AI inference into different containers to improve runtime stability and reproducibility

**Container A — Velodyne Driver (name: ros_velodyne)**
- Base Image: osrf/ros:noetic-desktop-full
- OS: Ubuntu 20.04
- ROS: Noetic
- Python: Included (ROS tools only, not used for ML)
- PyTorch: N/A

- Role:
  - Interface with Velodyne VLP-16 sensor
  - Receive raw UDP packets and publish PointCloud2 messages

- Publishes:
  - /velodyne_packets
  - /velodyne_points(sensor_msgs/PointCloud2)

- Notes:
  - Uses host networking (--net=host) for ROS communication
  - PointCloud2 fields include:
    x, y, z, intensity, ring, time (point_step = 22 bytes)

**Container B — 3D Detection (name: ppdet)**  
*ML stack compatibility (CUDA ↔ PyTorch ↔ OpenPCDet/spconv) must be consistent.*

- Base Image: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
- OS: Ubuntu 20.04
- ROS: Noetic (ros-noetic-ros-base)
- CUDA / cuDNN: CUDA 11.8 + cuDNN 8
- PyTorch: CUDA 11.8 wheel (cu118) + torchvision/torchaudio (cu118)
- Python: 3.8.x (Ubuntu 20.04 default)
- OpenPCDet: source build (commit: 233f849)  
  - Model: PointPillars (KITTI pretrained)
- spconv: spconv-cu118 (required for PointPillars)
- CUDA arch target (Ada): TORCH_CUDA_ARCH_LIST=8.9, FORCE_CUDA=1

- Role:
  - Real-time 3D object detection using PointPillars

- Subscribes:
  - /velodyne_points (sensor_msgs/PointCloud2)

- Publishes:
  - /ppdet/markers (visualization_msgs/MarkerArray)

> Note: The Dockerfile clones OpenPCDet from GitHub without pinning a specific commit.
> To ensure reproducibility, record the commit hash (233f849) or pin it via `git checkout <hash>` during build.



## System Architecture
<img width="1158" height="710" alt="image" src="https://github.com/user-attachments/assets/80c65eac-8b3f-4117-8a23-b25f5f143c37" />


## Quick start
> This Quickstart describes the minimal steps for running the system assuming a compatible GPU and driver environment.

- Create container A (ros_velodyne)
```
docker run -it -d \
  --name ros_velodyne \
  --net=host \
  --privileged \
  -e DISPLAY=:0 \
  -v /home/chu/PycharmProjects/PythonProject:/workspace/pycharm_scripts:rw \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  osrf/ros:noetic-desktop-full \
  /bin/bash
```





