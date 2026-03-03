#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from std_msgs.msg import Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np


# ================================
# 설정
# ================================
MODEL_PATH = (
    '/home/jaewookim/rokey_ws/src/yolo/resource/best.pt'
)

CLASS_AED = 'AED'
CLASS_PATIENT = 'Patient'
CLASS_CROWD = 'Crowd'
CLASS_RESPONDER = 'Responder'

YOLO_FPS = 10.0

# Depth 보정 파라미터
DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 5000
ROI_HALF_SIZE = 1          # 7x7 ROI
BOX_MARGIN_RATIO = 0.2     # 박스 가장자리 제거
EMA_ALPHA = 0.7            # depth 안정화 계수
# ================================


class YoloModelNode(Node):
    def __init__(self):
        super().__init__('yolo_model_node')

        # -------- namespace --------
        self.ns = self.get_namespace().replace('/', '')
        self.get_logger().info(f'[YOLO NODE] Namespace: {self.ns}')

        # -------- QoS --------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # -------- 기본 변수 --------
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.is_robot_a = None
        self.is_processing = False

        # Depth 필터 상태
        self.filtered_depth = None

        # -------- YOLO --------
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info('[YOLO NODE] Model loaded')

        # ================= PUBLISHERS =================
        # Robot A
        self.pub_aed = self.create_publisher(
            Bool, '/robotA/aed_detected', 10
        )
        self.pub_crowd = self.create_publisher(
            Bool, '/robotA/crowd_detected', 10
        )
        self.pub_patient_dist = self.create_publisher(
            Float32, '/robotA/patient/distance', 10
        )

        # Robot B
        self.pub_responder_done = self.create_publisher(
            Bool, '/robotB/responder_done', 10
        )

        # ================= SUBSCRIBERS =================
        self.create_subscription(
            Image,
            f'/{self.ns}/oakd/rgb/preview/image_raw',
            # f'/{self.ns}/oakd/rgb/image_raw/compressed',
            self.rgb_callback,
            qos
        )

        self.create_subscription(
            Image,
            f'/{self.ns}/oakd/stereo/image_raw',
            self.depth_callback,
            qos
        )

        self.create_subscription(
            Bool,
            '/robot_role',
            self.robot_role_callback,
            10
        )

        # -------- Timer --------
        self.timer = self.create_timer(
            1.0 / YOLO_FPS,
            self.run_yolo
        )

    # ================= CALLBACKS =================

    def robot_role_callback(self, msg: Bool):
        # 로그 출력
        self.get_logger().info(f'[YOLO NODE] Role signal received: {msg.data}')

        if self.ns == 'robot3':
            self.is_robot_a = msg.data
        elif self.ns == 'robot5':
            self.is_robot_a = not msg.data

        self.get_logger().info(f'{self.ns} is_robot_a: {self.is_robot_a}')



    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    # ================= DEPTH UTILS =================

    def get_stable_depth(self, x1, y1, x2, y2):
        """Bounding box 내부 ROI 평균 + EMA 적용"""

        h, w = self.latest_depth.shape

        # 박스 가장자리 제거
        bx1 = int(x1 + (x2 - x1) * BOX_MARGIN_RATIO)
        bx2 = int(x2 - (x2 - x1) * BOX_MARGIN_RATIO)
        by1 = int(y1 + (y2 - y1) * BOX_MARGIN_RATIO)
        by2 = int(y2 - (y2 - y1) * BOX_MARGIN_RATIO)

        if bx1 >= bx2 or by1 >= by2:
            return None

        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        # ROI 범위 제한
        x_min = max(cx - ROI_HALF_SIZE, 0)
        x_max = min(cx + ROI_HALF_SIZE + 1, w)
        y_min = max(cy - ROI_HALF_SIZE, 0)
        y_max = min(cy + ROI_HALF_SIZE + 1, h)

        roi = self.latest_depth[y_min:y_max, x_min:x_max]

        valid = roi[
            (roi > DEPTH_MIN_MM) &
            (roi < DEPTH_MAX_MM)
        ]

        if valid.size == 0:
            return None

        Z_raw = float(np.mean(valid)) / 1000.0

        # EMA 필터
        if self.filtered_depth is None:
            self.filtered_depth = Z_raw
        else:
            self.filtered_depth = (
                EMA_ALPHA * self.filtered_depth +
                (1.0 - EMA_ALPHA) * Z_raw
            )

        return self.filtered_depth

    # ================= YOLO LOGIC =================

    def run_yolo(self):
        if self.is_processing:
            return

        if (
            self.is_robot_a is None or
            self.latest_rgb is None or
            self.latest_depth is None
        ):
            return

        self.is_processing = True

        try:
            results = self.model(
                self.latest_rgb,
                conf=0.5,
                device=0,
                half=True,
                verbose=False
            )[0]

            aed_detected = False
            crowd_detected = False
            responder_done = False
            crowd_count = 0

            for box in results.boxes:
                cls_name = self.model.names[int(box.cls[0])]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_width = x2 - x1

                Z = self.get_stable_depth(x1, y1, x2, y2)
                if Z is None:
                    continue
                # 디버깅용: 클래스별 거리 출력
                # else:
                #     self.get_logger().info(
                #         f'[YOLO NODE] Detected {cls_name} at {Z:.2f} m'
                #     )

                # -------- Robot A --------
                if self.is_robot_a:
                    if cls_name == CLASS_AED:
                        aed_detected = True

                    elif cls_name == CLASS_CROWD:
                        if cls_name == 'Crowd' and box_width >= 20:
                            crowd_count += 1
                    elif cls_name == CLASS_PATIENT:
                        self.pub_patient_dist.publish(
                            Float32(data=Z)
                        )

                # -------- Robot B --------
                else:
                    if cls_name == CLASS_RESPONDER:
                        responder_done = True

            crowd_detected = (crowd_count >= 2)

            # -------- publish --------
            if self.is_robot_a:
                self.pub_aed.publish(Bool(data=aed_detected))
                self.pub_crowd.publish(Bool(data=crowd_detected))
            else:
                self.pub_responder_done.publish(
                    Bool(data=responder_done)
                )

        finally:
            self.is_processing = False


def main():
    rclpy.init()
    node = YoloModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()