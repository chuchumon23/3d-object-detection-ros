FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# ------------------------------------------------------------
# Base packages (build + python + common libs)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential cmake ninja-build \
    python3 python3-pip python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# ROS Noetic (Ubuntu 20.04 focal)
#  - include msg packages for PointCloud2 + RViz Marker publish
# ------------------------------------------------------------
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-rospy ros-noetic-rosbag \
    ros-noetic-tf2-ros ros-noetic-tf ros-noetic-tf2-tools \
    \
    # --- for PointCloud2 subscribe / bbox publish ---
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-visualization-msgs \
    \
    # --- python deps often needed by rospy packages ---
    python3-rospkg \
    python3-catkin-pkg \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# ------------------------------------------------------------
# PyTorch CUDA 11.8 wheel
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------
# OpenPCDet
# ------------------------------------------------------------
WORKDIR /root
RUN git clone https://github.com/open-mmlab/OpenPCDet.git
WORKDIR /root/OpenPCDet

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# spconv (cu118)  ✅ PointPillars 핵심
RUN python3 -m pip install --no-cache-dir spconv-cu118

# RTX 4060 laptop = sm_89 (Ada)
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV FORCE_CUDA="1"

# Build/install OpenPCDet (CUDA ops)
RUN python3 setup.py develop

WORKDIR /root/OpenPCDet
CMD ["/bin/bash"]
