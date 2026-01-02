# 3d-object-detection-ros-
This repository implements 3D object detection for vehicles using PointPillars AI algorithm, based on ROS and Docker. It processes 16-channel Velodyne point cloud data as input for real-time 3D vehicle detection.

# Overview
본 프로젝트는 16채널 Velodyne LiDAR(VLP-16)를 사용하여 포인트 클라우드 데이터를 수집하고,ROS 및 Docker 환경에서 PointPillars 기반 3D 객체 검출을 수행한다.

# Demo
<img width="473" height="434" alt="image" src="https://github.com/user-attachments/assets/60728fa1-c385-4f04-b890-f6b0199d8749" />

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
본 프로젝트는 센서 드라이버와 추론(AI)을 컨테이너로 분리하여, 실행 안정성 및 재현성을 높임

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





