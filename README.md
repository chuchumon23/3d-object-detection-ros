# 3d-object-detection-ros-
This repository implements 3D object detection for vehicles using PointPillars AI algorithm, based on ROS and Docker. It processes 16-channel Velodyne point cloud data as input for real-time 3D vehicle detection.

# Overview
이 프로젝트는 16채널 라이다로 센서 데이터(포인트클라우드)를 수집하고 ROS 및 Docker 환경에서 3D객체 검출 알고리즘인 Pointpillar을 활용하여 자동차 및 보행자를 3D객체 검출하는 프로젝트이다. 

# Demo
<img width="473" height="434" alt="image" src="https://github.com/user-attachments/assets/60728fa1-c385-4f04-b890-f6b0199d8749" />

# Environment & Requirements
## Hardware
GPU: NVIDIA RTX 4060 (Laptop)
Sensor: Velodyne VLP-16 (16-channel)

## Host(Laptop)
OS: Ubuntu 22.04 (running on external SSD)
Docker: used to isolate ROS/AI environments
Networking: containers communicate via ROS master (host or one container) / host networking recommended


본 프로젝트는 센서 드라이버와 추론(AI)을 컨테이너로 분리하여, 실행 안정성 및 재현성을 높임

- GPU /CUDA : RTX 4060
- OS/ROS version : 22.04 UBUNTU (HOST)
  Docker: container A(Velodyne driver node/20.04 UBUNTU, ROS NOETIC), container B(3d object detection node/20.04 UBUNTU, ROS NOETIC)
- Pytorch version: 
- python version:
- OpenPCdet version: 
- 센서 모델 : velodyne 16channel
  
ML(machine learning) 스택 궁합을 잘 고려해야함



