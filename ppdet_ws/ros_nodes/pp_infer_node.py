#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================
# av2 dummy module (IMPORT HACK)
# ==========================================
import sys
import types

av2 = types.ModuleType("av2")
av2.utils = types.ModuleType("av2.utils")
av2.utils.io = types.ModuleType("av2.utils.io")

def _dummy(*args, **kwargs):
    raise RuntimeError("av2 is not available (dummy module)")

av2.utils.io.read_feather = _dummy

sys.modules["av2"] = av2
sys.modules["av2.utils"] = av2.utils
sys.modules["av2.utils.io"] = av2.utils.io

# ==========================================
# Standard imports
# ==========================================
import os
import numpy as np
import rospy
import torch

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
import tf.transformations as tft

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ==========================================
# Minimal Dataset wrapper
# ==========================================
class ROSDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, logger):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=None,
            logger=logger
        )

    def get_one(self, points_np: np.ndarray):
        data_dict = self.prepare_data(data_dict={"points": points_np})
        return data_dict


# ==========================================
# Utils
# ==========================================
def pc2_to_xyzi(msg: PointCloud2) -> np.ndarray:
    """
    PointCloud2 -> (N,4) float32 [x,y,z,intensity]
    ring/time 등 추가 필드가 있어도 field_names로 안전하게 추출됨.
    """
    pts = np.asarray(
        list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
        dtype=np.float32
    )
    if pts.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    # ========== CRITICAL FIX: Intensity 정규화 ==========
    # Velodyne 실제 출력: 0~255 범위
    # KITTI 학습 데이터: 0~1 범위
    # PointPillars 모델은 0~1 범위를 기대함
    if pts.shape[0] > 0 and pts[:, 3].max() > 1.0:
        pts[:, 3] = pts[:, 3] / 255.0
    # ====================================================
    
    return pts


def quat_from_yaw(yaw: float) -> Quaternion:
    q = tft.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def class_color(cls_name: str):
    if cls_name == "Car":
        return (0.2, 1.0, 0.2)
    if cls_name == "Pedestrian":
        return (1.0, 0.2, 0.2)
    if cls_name == "Cyclist":
        return (0.2, 0.2, 1.0)
    return (1.0, 1.0, 1.0)


# ==========================================
# Node
# ==========================================
class PointPillarsROS:
    def __init__(self):
        rospy.loginfo("Initializing PointPillars ROS node...")

        tools_dir = "/root/OpenPCDet/tools"
        if os.path.isdir(tools_dir):
            os.chdir(tools_dir)
        rospy.loginfo(f"Working directory: {os.getcwd()}")

        # Params
        self.topic_in = rospy.get_param("~topic_in", "/velodyne_points")
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.score_thr = float(rospy.get_param("~score_thr", 0.3))

        self.only_car = bool(rospy.get_param("~only_car", False))
        self.detect_pedestrian = bool(rospy.get_param("~detect_pedestrian", True))
        self.detect_cyclist = bool(rospy.get_param("~detect_cyclist", True))

        # 필터링 파라미터
        self.max_distance = float(rospy.get_param("~max_distance", 40.0))
        self.max_box_size = rospy.get_param("~max_box_size", [5.5, 2.5, 2.5])  # [x, y, z]
        self.min_box_size = rospy.get_param("~min_box_size", [1.5, 0.8, 0.5])  # [x, y, z]

        # z 기준: KITTI(OpenPCDet) 박스 z가 bottom일 가능성이 높아서 center로 보정
        self.z_is_bottom = bool(rospy.get_param("~z_is_bottom", True))

        self.cfg_file = rospy.get_param(
            "~cfg_file",
            "/root/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
        )
        self.ckpt = rospy.get_param(
            "~ckpt",
            "/root/OpenPCDet/pretrained/pointpillar_7728.pth"
        )

        rospy.loginfo(f"Config: {self.cfg_file}")
        rospy.loginfo(f"Checkpoint: {self.ckpt}")
        rospy.loginfo(f"Subscribe: {self.topic_in}")
        rospy.loginfo(f"Score thr: {self.score_thr}")
        rospy.loginfo(f"Max distance: {self.max_distance}m")
        rospy.loginfo(f"Box size range: {self.min_box_size} ~ {self.max_box_size}")
        rospy.loginfo(f"z_is_bottom: {self.z_is_bottom}")

        # Logger + Config
        self.logger = common_utils.create_logger()
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.class_names = cfg.CLASS_NAMES
        rospy.loginfo(f"Classes: {self.class_names}")

        # Dataset
        self.dataset = ROSDataset(cfg.DATA_CONFIG, self.class_names, logger=self.logger)

        # Model
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(self.class_names),
            dataset=self.dataset
        )
        self.model.cuda()
        self.model.eval()
        self.model.load_params_from_file(
            filename=self.ckpt,
            logger=self.logger,
            to_cpu=False
        )

        # Pub/Sub
        self.pub = rospy.Publisher("/ppdet/markers", MarkerArray, queue_size=1)
        self.sub = rospy.Subscriber(
            self.topic_in, PointCloud2, self.cb,
            queue_size=1, buff_size=2**24
        )

        self.frame_count = 0
        rospy.loginfo("PointPillars ROS node ready.")

    def _class_allowed(self, cls_name: str) -> bool:
        if self.only_car and cls_name != "Car":
            return False
        if cls_name == "Pedestrian" and not self.detect_pedestrian:
            return False
        if cls_name == "Cyclist" and not self.detect_cyclist:
            return False
        return True

    @torch.no_grad()
    def cb(self, msg: PointCloud2):
        self.frame_count += 1

        points = pc2_to_xyzi(msg)
        
        # ========== 진단 로그 ==========
        if self.frame_count == 1 or self.frame_count % 10 == 0:
            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"[Frame {self.frame_count}] 포인트 개수: {points.shape[0]}")
            if points.shape[0] > 0:
                rospy.loginfo(f"  X 범위: [{points[:,0].min():8.2f}, {points[:,0].max():8.2f}]")
                rospy.loginfo(f"  Y 범위: [{points[:,1].min():8.2f}, {points[:,1].max():8.2f}]")
                rospy.loginfo(f"  Z 범위: [{points[:,2].min():8.2f}, {points[:,2].max():8.2f}]")
                rospy.loginfo(f"  I 범위: [{points[:,3].min():8.2f}, {points[:,3].max():8.2f}]")
            rospy.loginfo(f"{'='*60}")
        # ===============================

        if points.shape[0] < 50:
            return

        data_dict = self.dataset.get_one(points)
        batch = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(batch)

        pred_dicts, _ = self.model.forward(batch)
        pred = pred_dicts[0]

        boxes = pred["pred_boxes"].detach().cpu().numpy()    # (M,7)
        scores = pred["pred_scores"].detach().cpu().numpy()
        labels = pred["pred_labels"].detach().cpu().numpy()

        frame_id = self.frame_id_override if self.frame_id_override else msg.header.frame_id

        ma = MarkerArray()
        mid = 0
        published = 0
        filtered_by_score = 0
        filtered_by_distance = 0
        filtered_by_size = 0

        for box, score, label in zip(boxes, scores, labels):
            # Score threshold
            if score < self.score_thr:
                filtered_by_score += 1
                continue

            cls_name = self.class_names[int(label) - 1]
            if not self._class_allowed(cls_name):
                continue

            x, y, z, dx, dy, dz, yaw = box.tolist()
            
            # ========== 필터링 추가 ==========
            # 1. 거리 필터 (너무 먼 물체 제외)
            distance = np.sqrt(x**2 + y**2)
            if distance > self.max_distance:
                filtered_by_distance += 1
                continue
            
            # 2. 크기 필터 (비정상적으로 큰/작은 물체 제외)
            if (dx > self.max_box_size[0] or 
                dy > self.max_box_size[1] or 
                dz > self.max_box_size[2]):
                filtered_by_size += 1
                continue
            
            if (dx < self.min_box_size[0] or 
                dy < self.min_box_size[1] or 
                dz < self.min_box_size[2]):
                filtered_by_size += 1
                continue
            # ================================
            
            # ========== 박스 진단 로그 ==========
            if self.frame_count == 1 or self.frame_count % 10 == 0:
                rospy.loginfo(f"  → {cls_name}: pos=({x:6.1f}, {y:6.1f}, {z:6.1f}), "
                             f"size=({dx:4.1f}, {dy:4.1f}, {dz:4.1f}), "
                             f"dist={distance:5.1f}m, yaw={yaw:5.2f}, score={score:.2f}")
            # ===================================

            # RViz Marker.CUBE는 pose가 "박스 중심"이므로,
            # KITTI류에서 z가 bottom 기준이면 center로 보정해줘야 함.
            z_marker = z
            if self.z_is_bottom:
                z_marker = z + dz / 2.0

            # ========== Bounding Box Marker ==========
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = msg.header.stamp
            m.ns = "ppdet_bbox"
            m.id = mid
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z_marker
            m.pose.orientation = quat_from_yaw(yaw)

            m.scale.x = dx
            m.scale.y = dy
            m.scale.z = dz

            r, g, b = class_color(cls_name)
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.6
            m.lifetime = rospy.Duration(0.1)

            ma.markers.append(m)
            mid += 1

            # ========== Text Marker (클래스명 + Score) ==========
            text_m = Marker()
            text_m.header.frame_id = frame_id
            text_m.header.stamp = msg.header.stamp
            text_m.ns = "ppdet_text"
            text_m.id = mid
            text_m.type = Marker.TEXT_VIEW_FACING
            text_m.action = Marker.ADD

            # 텍스트 위치: 박스 상단 위 0.5m
            text_m.pose.position.x = x
            text_m.pose.position.y = y
            text_m.pose.position.z = z_marker + dz/2.0 + 0.5
            text_m.pose.orientation.w = 1.0

            # 텍스트 내용
            text_m.text = f"{cls_name}\n{score:.2f}"
            
            # 텍스트 크기
            text_m.scale.z = 0.5  # 텍스트 높이

            # 텍스트 색상 (박스와 동일)
            text_m.color.r = r
            text_m.color.g = g
            text_m.color.b = b
            text_m.color.a = 1.0
            text_m.lifetime = rospy.Duration(0.1)

            ma.markers.append(text_m)
            mid += 1

            published += 1

        if published > 0:
            self.pub.publish(ma)

        if self.frame_count % 10 == 0:
            rospy.loginfo(f"[Frame {self.frame_count}] 발행된 검출 결과: {published}개")
            rospy.loginfo(f"  필터링됨: score={filtered_by_score}, "
                         f"distance={filtered_by_distance}, size={filtered_by_size}\n")


# ==========================================
if __name__ == "__main__":
    rospy.init_node("pp_infer_node", anonymous=False)
    PointPillarsROS()
    rospy.spin()
