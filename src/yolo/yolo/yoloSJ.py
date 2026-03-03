#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2

class DualYoloNode(Node):
    def __init__(self):
        super().__init__('dual_yolo_node')
        self.bridge = CvBridge()
        self.model = YOLO('/home/jaewookim/rokey_ws/src/yolo/resource/best.pt')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # -------- 실시간 역할 상태 (표 기준 초기화) --------
        # True: Robot3=A, Robot5=B / False: Robot3=B, Robot5=A
        self.current_role_is_true = None

        self.data = {
            'robot3': {'rgb': None, 'depth': None},
            'robot5': {'rgb': None, 'depth': None}
        }

        # ================= PUBLISHERS =================
        self.pub_aed = self.create_publisher(Bool, '/robotA/aed_detected', 10)
        self.pub_crowd_a = self.create_publisher(Bool, '/robotA/crowd_detected', 10)
        self.pub_patient_dist = self.create_publisher(Float32, '/robotA/patient/distance', 10)

        self.pub_responder_done = self.create_publisher(Bool, '/robotB/responder_done', 10)
        self.pub_crowd_b = self.create_publisher(Bool, '/robotB/crowd_detected', 10)

        # ================= SUBSCRIBERS =================
        # Robot3, Robot5 동시 구독
        for ns in ['robot3', 'robot5']:
            self.create_subscription(Image, f'/{ns}/oakd/rgb/preview/image_raw',
                                     lambda msg, n=ns: self.rgb_cb(msg, n), qos)
            self.create_subscription(Image, f'/{ns}/oakd/stereo/image_raw',
                                     lambda msg, n=ns: self.depth_cb(msg, n), qos)

        # 역할 신호 구독
        self.create_subscription(Bool, '/robot_role', self.role_callback, 10)

        self.create_timer(0.1, self.run_yolo_dual)

    def role_callback(self, msg: Bool):
        self.current_role_is_true = msg.data
        if self.current_role_is_true:
            self.get_logger().info(":arrows_counterclockwise: [Role] True 수신: Robot3=A(AED), Robot5=B(EMT)")
        else:
            self.get_logger().info(":arrows_counterclockwise: [Role] False 수신: Robot3=B(EMT), Robot5=A(AED)")

    def rgb_cb(self, msg, robot_id):
        self.data[robot_id]['rgb'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_cb(self, msg, robot_id):
        self.data[robot_id]['depth'] = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def run_yolo_dual(self):
        # 역할 신호가 들어오기 전까지 대기
        if self.current_role_is_true is None:
            return

        for robot_id in ['robot3', 'robot5']:
            rb_data = self.data[robot_id]
            if rb_data['rgb'] is None or rb_data['depth'] is None:
                continue

            # :star: 표 로직 적용
            # True일 때: Robot3가 A / False일 때: Robot5가 A
            if self.current_role_is_true:
                is_currently_a = (robot_id == 'robot3')
            else:
                is_currently_a = (robot_id == 'robot5')

            results = self.model(rb_data['rgb'], conf=0.4, device=0, half=True, verbose=False)[0]

            aed_found = False; responder_found = False; crowd_count = 0
            debug_frame = rb_data['rgb'].copy()

            for box in results.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                depth_val = rb_data['depth'][cy, cx] if cy < rb_data['depth'].shape[0] else 0
                Z = depth_val / 1000.0 if depth_val > 0 else 0.0

                # 시각화 (A는 파랑, B는 노랑)
                color = (255, 0, 0) if is_currently_a else (0, 255, 255)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_frame, f"{cls_name} {Z:.1f}m", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # if cls_name == 'Crowd':
                #     crowd_count += 1

                # cls_name이 'Crowd'이고, 해당 물체의 depth가 1.75m 이하인 경우에만 카운트
                if cls_name == 'Crowd' and 0 < Z <= 1.75:
                    crowd_count += 1

                if is_currently_a:
                    if cls_name == 'AED': aed_found = True
                    elif cls_name == 'Patient': self.pub_patient_dist.publish(Float32(data=float(Z)))
                else:
                    if cls_name == 'Responder': responder_found = True

            crowd_status = (crowd_count >= 2)

            # --- 결과 발행 ---
            if is_currently_a:
                self.pub_aed.publish(Bool(data=aed_found))
                self.pub_crowd_a.publish(Bool(data=crowd_status))
            else:
                self.pub_responder_done.publish(Bool(data=responder_found))
                self.pub_crowd_b.publish(Bool(data=crowd_status))

            # 화면에 정보 표시
            role_text = "Robot A (AED)" if is_currently_a else "Robot B (EMT)"
            cv2.putText(debug_frame, f"{robot_id}: {role_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(f"Monitoring {robot_id}", debug_frame)

        cv2.waitKey(1)

def main():
    rclpy.init()
    rclpy.spin(DualYoloNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()