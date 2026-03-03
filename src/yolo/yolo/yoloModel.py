#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from std_msgs.msg import Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ================================
# 설정
# ================================
# 환경에 맞추어 반드시 변경하십시오.
MODEL_PATH = (
    '/home/jaewookim/rokey_ws/src/yolo/resource/best.pt'
)

CLASS_AED = 'AED'
CLASS_PATIENT = 'Patient'
CLASS_CROWD = 'Crowd'
CLASS_RESPONDER = 'Responder'

YOLO_FPS = 10.0   # 실전 권장 8~12
# ================================


class YoloModelNode(Node):
    def __init__(self):
        super().__init__('yolo_model_node')

        # -------- namespace --------
        self.ns = self.get_namespace().replace('/', '')
        self.get_logger().info(f'[YOLO NODE] Namespace: {self.ns}')

        # -------- QoS (프레임 밀림 방지) --------
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
            f'/{self.ns}/oakd/rgb/preview/image_raw',   # ⭐ preview 사용
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
        # 초당 10번만 수행
        self.timer = self.create_timer(
            1.0 / YOLO_FPS,
            self.run_yolo
        )

    # ================= CALLBACKS =================

    def robot_role_callback(self, msg: Bool):
        # 로그 생성
        if msg.data:
            self.get_logger().info(
                ":arrows_counterclockwise: [Role] True 수신: Robot=A(AED)"
            )
        else:
            self.get_logger().info(
                ":arrows_counterclockwise: [Role] False 수신: Robot=B(EMT)"
            )
        self.is_robot_a = msg.data

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_callback(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

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
                conf=0.4,
                device=0,     # GPU 고정
                half=True,    # FP16
                verbose=False
            )[0]

            aed_detected = False
            crowd_detected = False
            responder_done = False

            for box in results.boxes:
                cls_name = self.model.names[int(box.cls[0])]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if (
                    cx < 0 or cy < 0 or
                    cx >= self.latest_depth.shape[1] or
                    cy >= self.latest_depth.shape[0]
                ):
                    continue

                depth_mm = self.latest_depth[cy, cx]
                if depth_mm <= 0:
                    continue

                Z = depth_mm / 1000.0

                # -------- Robot A --------
                if self.is_robot_a:
                    if cls_name == CLASS_AED:
                        aed_detected = True
                    elif cls_name == CLASS_CROWD:
                        crowd_detected = True
                    elif cls_name == CLASS_PATIENT:
                        self.pub_patient_dist.publish(Float32(data=Z))

                # -------- Robot B --------
                else:
                    if cls_name == CLASS_RESPONDER:
                        responder_done = True

            # -------- publish --------
            if self.is_robot_a:
                self.pub_aed.publish(Bool(data=aed_detected))
                self.pub_crowd.publish(Bool(data=crowd_detected))
            else:
                self.pub_responder_done.publish(Bool(data=responder_done))

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
