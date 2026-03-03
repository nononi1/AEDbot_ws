#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from std_msgs.msg import Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

class DualYoloNode(Node):
    def __init__(self):
        super().__init__('dual_yolo_node')
        self.bridge = CvBridge()
        
        # 모델 로드 (verbose=False로 불필요한 콘솔 출력 제거)
        self.model = YOLO('/home/jaewookim/rokey_ws/src/yolo/resource/best.pt')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        # -------- 실시간 역할 상태 --------
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
        for ns in ['robot3', 'robot5']:
            self.create_subscription(Image, f'/{ns}/oakd/rgb/preview/image_raw',
                                     lambda msg, n=ns: self.rgb_cb(msg, n), qos)
            self.create_subscription(Image, f'/{ns}/oakd/stereo/image_raw',
                                     lambda msg, n=ns: self.depth_cb(msg, n), qos)

        self.create_subscription(Bool, '/robot_role', self.role_callback, 10)

        # 타이머 주기: 필요에 따라 조절 (0.1s = 10Hz)
        self.create_timer(0.1, self.run_yolo_dual)

    def role_callback(self, msg: Bool):
        prev_role = self.current_role_is_true
        self.current_role_is_true = msg.data
        if prev_role != self.current_role_is_true:
            role_str = "True (R3=A, R5=B)" if self.current_role_is_true else "False (R3=B, R5=A)"
            self.get_logger().info(f"[Role Updated] {role_str}")

    def rgb_cb(self, msg, robot_id):
        # UI 표시가 없으므로 bgr8 대신 모델 입력에 맞는 포맷으로 변환해도 좋으나,
        # YOLO는 내부적으로 처리하므로 편의상 유지
        try:
            self.data[robot_id]['rgb'] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB Error: {e}")

    def depth_cb(self, msg, robot_id):
        try:
            self.data[robot_id]['depth'] = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth Error: {e}")

    def run_yolo_dual(self):
        # 역할 신호 대기
        if self.current_role_is_true is None:
            return

        # 1. 추론할 이미지와 해당 로봇 ID 수집 (Batch 준비)
        valid_robots = []
        batch_images = []

        for robot_id in ['robot3', 'robot5']:
            rb_data = self.data[robot_id]
            if rb_data['rgb'] is not None and rb_data['depth'] is not None:
                valid_robots.append(robot_id)
                batch_images.append(rb_data['rgb'])

        if not batch_images:
            return

        # 2. Batch Inference (한 번의 호출로 여러 이미지 처리 -> 속도 향상 핵심)
        # verbose=False로 로그 출력 오버헤드 제거, half=True로 FP16 가속
        results = self.model(batch_images, conf=0.4, device=0, half=True, verbose=False)

        # 3. 결과 처리
        for i, robot_id in enumerate(valid_robots):
            result = results[i] # 결과 리스트는 입력 리스트 순서와 동일함
            depth_map = self.data[robot_id]['depth']
            
            # 현재 로봇이 A 역할인지 확인
            if self.current_role_is_true:
                is_currently_a = (robot_id == 'robot3')
            else:
                is_currently_a = (robot_id == 'robot5')

            aed_found = False
            responder_found = False
            crowd_count = 0

            # 박스 데이터 처리 (CPU로 이동 및 numpy 변환)
            # result.boxes.data: [x1, y1, x2, y2, conf, cls]
            if result.boxes:
                boxes = result.boxes
                cls_indices = boxes.cls.cpu().numpy().astype(int)
                xyxys = boxes.xyxy.cpu().numpy().astype(int)
                
                for cls_idx, box in zip(cls_indices, xyxys):
                    cls_name = self.model.names[cls_idx]
                    
                    if cls_name == 'Crowd':
                        crowd_count += 1
                    
                    # 로봇 역할에 따라 필요한 연산만 수행
                    if is_currently_a:
                        if cls_name == 'AED':
                            aed_found = True
                        elif cls_name == 'Patient':
                            # 거리 계산 (중심점)
                            x1, y1, x2, y2 = box
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            # Depth map 범위 체크 (Safety)
                            h, w = depth_map.shape
                            if 0 <= cy < h and 0 <= cx < w:
                                depth_val = depth_map[cy, cx]
                                if depth_val > 0:
                                    Z = float(depth_val) / 1000.0
                                    self.pub_patient_dist.publish(Float32(data=Z))
                    else:
                        # Robot B (EMT)
                        if cls_name == 'Responder':
                            responder_found = True

            crowd_status = (crowd_count >= 2)

            # --- 결과 발행 ---
            if is_currently_a:
                self.pub_aed.publish(Bool(data=aed_found))
                self.pub_crowd_a.publish(Bool(data=crowd_status))
            else:
                self.pub_responder_done.publish(Bool(data=responder_found))
                self.pub_crowd_b.publish(Bool(data=crowd_status))

def main():
    rclpy.init()
    # 멀티스레드 Executor를 사용하면 콜백 병목을 더 줄일 수 있으나, 
    # 현재 로직에서는 단순 Spin으로도 충분히 빠릅니다.
    node = DualYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()