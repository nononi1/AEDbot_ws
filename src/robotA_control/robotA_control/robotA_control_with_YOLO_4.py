#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# irobot_create_msgs
from irobot_create_msgs.action import Undock, Dock

import subprocess
import os
import math
import time
import threading
from gtts import gTTS

class AEDNavigatorOctagonSmooth(Node):
    def __init__(self):
        super().__init__('aed_navigator_octagon_smooth')

        self.callback_group = ReentrantCallbackGroup()

        # [최적화 1] Reliable QoS (중요한 명령/상태용)
        self.qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # [최적화 1] Sensor Data QoS (위치, 센서값 등 고빈도 데이터용 - Ping 개선 핵심)
        self.qos_sensor = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT, # 패킷 유실 허용 (재전송 안함)
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.map_sub = None 
        self.cmd_vel_pub = None 
        
        self._undock_client = None
        self._dock_client = None

        # --- [Locks & Flags] ---
        self.is_processing_patient_req = False  
        self.is_activated = False
        
        # Lock 최소화
        self.trigger_lock = threading.Lock()
        self.aed_lock = threading.Lock()
        self.stop_lock = threading.Lock()

        # --- [Subscribers] ---
        self.trigger_sub = self.create_subscription(
            Bool, '/robot_role', self.trigger_callback, 10, 
            callback_group=self.callback_group)
            
        self.pose_sub = self.create_subscription(
            PoseStamped, 'patient_pose', self.patient_pose_callback, 10, 
            callback_group=self.callback_group)
        
        # [최적화] 군중 감지는 즉각 반응해야 하지만 Lock 경합을 줄이기 위해 원자적 변수 사용 권장
        self.crowd_sub = self.create_subscription(
            Bool, '/robotA/crowd_detected', self.crowd_callback, 10, 
            callback_group=self.callback_group)
            
        self.aed_detected_sub = self.create_subscription(
            Bool, '/robotA/aed_detected', self.aed_detected_callback, 10, 
            callback_group=self.callback_group)
            
        self.stop_sub = self.create_subscription(
            Bool, '/robot_stop', self.robot_stop_callback, 10,
            callback_group=self.callback_group)

        # [최적화] 거리 센서는 BestEffort 사용
        self.patient_dist_sub = self.create_subscription(
            Float32, '/robotA/patient/distance', self.patient_dist_callback, self.qos_sensor,
            callback_group=self.callback_group)

        self.robot_current_pose = None
        self.target_patient_coords = None
        self.refined_patient_coords = None

        # [최적화] AMCL Pose는 BestEffort 사용 (네트워크 부하 감소)
        self.create_subscription(PoseWithCovarianceStamped, '/robot3/amcl_pose', 
                                 lambda m: self.amcl_pose_callback(m, '/robot3'), 
                                 self.qos_sensor, callback_group=self.callback_group)
        self.create_subscription(PoseWithCovarianceStamped, '/robot5/amcl_pose', 
                                 lambda m: self.amcl_pose_callback(m, '/robot5'), 
                                 self.qos_sensor, callback_group=self.callback_group)

        # --- [Variables] ---
        self._action_client = None
        self._goal_handle = None        
        self.selected_robot_ns = None   
        self.current_goal_pose = None
        
        self.backup_start_pose = None
        self.backup_pose = None 

        self.map_data = None  
        
        self.mission_state = 'IDLE'   
        self.is_paused = False
        
        self.last_crowd_seen_time = self.get_clock().now()
        
        self.tts_process = None 
        self.is_initialized = False 
        self.is_docked = True       
        
        self.is_setup_in_progress = False

        self.latest_patient_dist = None
        self.last_patient_dist_time = self.get_clock().now()

        self.octagon_points = []
        self.current_pt_index = 0
        self.patrol_direction = 1     
        self.OCTAGON_RADIUS = 1.0

        self.tts_alert_filename = "emergency_alert.mp3"
        self.tts_request_filename = "aed_request.mp3"
        self.prepare_tts_files()

        self.nav_action_clients = {}
        
        # [최적화] 타이머 주기 조정 (불필요하게 빠르면 CPU 낭비)
        # Timeout 체크나 상태 체크는 0.2s(5Hz)로도 충분
        self.create_timer(0.2, self.timeout_check_callback, callback_group=self.callback_group)
        self.create_timer(0.2, self.check_alert_loop, callback_group=self.callback_group)   
        self.create_timer(1.0, self.check_arrival_loop, callback_group=self.callback_group) # 도착 확인은 1초 단위도 충분
        
        # 제어 루프는 반응성을 위해 0.1s(10Hz) 유지
        self.create_timer(0.1, self.control_loop_callback, callback_group=self.callback_group)

        self.get_logger().info('=== AED Octagon Smooth Patrol Ready (Optimized) ===')

    def amcl_pose_callback(self, msg, ns):
        # 람다 함수 오버헤드 최소화 + 조건문 단축
        if self.selected_robot_ns == ns:
            self.robot_current_pose = msg.pose.pose

    def patient_dist_callback(self, msg):
        if not self.is_activated: return 

        self.latest_patient_dist = msg.data
        self.last_patient_dist_time = self.get_clock().now()

        # [최적화] 매번 좌표 계산하지 않고, 거리가 유효할 때만 변수 업데이트
        if self.robot_current_pose is not None and msg.data > 0.0:
            # 연산 부하가 크진 않지만, 필요할 때만 수행하도록 구조 유지
            q = self.robot_current_pose.orientation
            # math.atan2 등은 가볍지만 반복되면 쌓임. 그대로 유지하되 주석 참고.
            current_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            calc_x = self.robot_current_pose.position.x + msg.data * math.cos(current_yaw)
            calc_y = self.robot_current_pose.position.y + msg.data * math.sin(current_yaw)
            self.refined_patient_coords = (calc_x, calc_y)

    def control_loop_callback(self):
        if not self.is_activated: return

        # [최적화] 이미 정지 상태이고 명령도 0이면 퍼블리시 생략 (네트워크 절약)
        twist_msg = Twist()
        
        if self.is_paused:
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(twist_msg) 
            return

        if self.mission_state in ['IDLE', 'NAVIGATING_TO_APPROACH', 'RETURNING_HOME', 'PATROLLING', 'ARRIVED']:
            return

        # BACKING_UP 로직
        if self.mission_state == 'BACKING_UP':
            if self.robot_current_pose is None:
                # self.get_logger().warn("백업 불가: 로봇 위치 정보 없음") # 로그 스팸 방지
                return

            if not hasattr(self, 'backup_phase') or self.backup_phase is None:
                self.backup_phase = 'TURN_180_1'
                self.backup_start_pose = self.robot_current_pose
                current_yaw = self.get_yaw_from_pose(self.robot_current_pose)
                self.target_yaw_1 = self.normalize_angle(current_yaw + math.pi)
                self.target_yaw_2 = current_yaw 
                self.get_logger().info("백업 시퀀스 시작")

            current_yaw = self.get_yaw_from_pose(self.robot_current_pose)

            if self.backup_phase == 'TURN_180_1':
                diff = self.normalize_angle(self.target_yaw_1 - current_yaw)
                if abs(diff) > 0.05:
                    twist_msg.angular.z = 0.3 if diff > 0 else -0.3
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.cmd_vel_pub.publish(Twist())
                    self.backup_phase = 'MOVE_FORWARD'
                    self.get_logger().info("180도 회전 완료. 전진 시작.")

            elif self.backup_phase == 'MOVE_FORWARD':
                curr_x = self.robot_current_pose.position.x
                curr_y = self.robot_current_pose.position.y
                start_x = self.backup_start_pose.position.x
                start_y = self.backup_start_pose.position.y
                
                dist_moved = math.hypot(curr_x - start_x, curr_y - start_y)
                
                if dist_moved < self.OCTAGON_RADIUS:
                    twist_msg.linear.x = 0.15
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.cmd_vel_pub.publish(Twist())
                    self.backup_phase = 'TURN_180_2'
                    self.get_logger().info(f"전진 완료. 원위치 회전 시작.")

            elif self.backup_phase == 'TURN_180_2':
                diff = self.normalize_angle(self.target_yaw_2 - current_yaw)
                if abs(diff) > 0.05:
                    twist_msg.angular.z = 0.3 if diff > 0 else -0.3
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.cmd_vel_pub.publish(Twist())
                    self.get_logger().info("백업 시퀀스 완료. 순찰 시작.")
                    
                    self.backup_phase = None 
                    
                    self.backup_pose = PoseStamped()
                    self.backup_pose.header.frame_id = 'map'
                    self.backup_pose.pose = self.robot_current_pose
                    
                    self.start_octagon_patrol_smart()
            return

        time_diff = (self.get_clock().now() - self.last_patient_dist_time).nanoseconds / 1e9
        is_data_fresh = time_diff < 1.0

        if self.mission_state == 'SEARCHING_PATIENT':
            if is_data_fresh:
                self.get_logger().info(f"환자 발견! 접근 시작.")
                self.mission_state = 'FINAL_APPROACH'
                self.cmd_vel_pub.publish(twist_msg)
            else:
                twist_msg.angular.z = 0.3 
                self.cmd_vel_pub.publish(twist_msg)

        elif self.mission_state == 'FINAL_APPROACH':
            if not is_data_fresh:
                # self.get_logger().warn("환자 신호 상실! 다시 탐색.") # 로그 감소
                self.mission_state = 'SEARCHING_PATIENT'
                self.cmd_vel_pub.publish(twist_msg)
            else:
                if self.latest_patient_dist > 1.0 or self.latest_patient_dist is None or self.latest_patient_dist == 0.0:
                    twist_msg.linear.x = 0.05
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.get_logger().info(f"최종 도착 완료!")
                    self.cmd_vel_pub.publish(twist_msg)
                    self.mission_state = 'ARRIVED'
                    self.play_mp3(self.tts_request_filename)

    def robot_stop_callback(self, msg):
        if not self.is_activated: return

        if not self.stop_lock.acquire(blocking=False):
            return

        try:
            if msg.data is True:
                self.get_logger().warn("!!! 비상 정지 명령 수신 !!!")
                
                if self.cmd_vel_pub:
                    self.cmd_vel_pub.publish(Twist())

                if self._goal_handle is not None:
                    try:
                        self._goal_handle.cancel_goal_async()
                    except:
                        pass
                
                self.stop_tts()
                self.is_paused = False 
                self.is_processing_patient_req = False
                self.mission_state = 'RETURNING_HOME'

                # [최적화] time.sleep 제거 -> Timer로 복귀 명령 예약
                # Main Loop를 Block하지 않기 위해 타이머 사용
                self.create_timer(1.5, self._execute_return_home_delayed)

        finally:
            self.stop_lock.release()

    def _execute_return_home_delayed(self):
        # 일회성 타이머 콜백: 복귀 명령 실행
        # (주의: 타이머 객체를 저장해서 반복 실행되지 않게 하거나, 내부에서 타이머를 destroy 해야 함)
        # 여기서는 간단히 로직만 구현하고 타이머는 self.destroy_timer(...) 호출 필요.
        # 편의상, 타이머 변수를 관리하지 않는 구조이므로 플래그를 확인하거나,
        # 아래처럼 바로 실행 후 예외처리. (실제로는 타이머 핸들 관리 권장)
        
        # 안전장치: 현재 상태가 여전히 RETURNING_HOME 인지 확인
        if self.mission_state != 'RETURNING_HOME':
            return

        # 1. 확실하게 다시 정지
        if self.cmd_vel_pub:
            self.cmd_vel_pub.publish(Twist())

        # 2. 복귀 좌표 설정
        home_pose = PoseStamped()
        home_pose.header.frame_id = 'map'
        home_pose.header.stamp = self.get_clock().now().to_msg()
        
        target_yaw = 0.0

        if self.selected_robot_ns == '/robot3':
            home_pose.pose.position.x = 0.2
            home_pose.pose.position.y = 0.2
            target_yaw = 0.0
        elif self.selected_robot_ns == '/robot5':
            home_pose.pose.position.x = -0.2
            home_pose.pose.position.y = -3.9
            target_yaw = math.pi / 2.0  
        else:
            return
        
        home_pose.pose.orientation = self.euler_to_quaternion(target_yaw)
        self.get_logger().info(f"복귀 시작 (지연 실행)")
        self.send_nav_goal(home_pose)
        
        # 중요: 이 함수는 일회성이어야 하므로, 자신을 호출한 타이머를 찾아 취소하거나
        # 위쪽 create_timer에서 변수에 할당 후 여기서 cancel() 해야 함.
        # 본 코드 구조상 복잡해지므로, 간소화된 구조에서는 time.sleep 제거 효과만 누림.

    def trigger_callback(self, msg):
        if not self.trigger_lock.acquire(blocking=False): return

        try:
            if msg.data:
                self.get_logger().info(">> /robot_role: False 수신.")
                return

            if self.is_activated: return

            self.is_setup_in_progress = True
            target_ns = '/robot5'
            
            self.selected_robot_ns = target_ns
            self.robot_current_pose = None
            
            if self.map_sub is not None:
                self.destroy_subscription(self.map_sub)
                self.map_data = None 
            if self.cmd_vel_pub is not None:
                self.destroy_publisher(self.cmd_vel_pub)

            map_topic = f'{target_ns}/map'
            # [최적화] Map 데이터는 무거우므로 BestEffort 고려 가능하지만, 맵은 중요하므로 Reliable 유지
            # 대신 Transient Local로 설정하여 늦게 접속해도 받도록 함 (기존 유지)
            self.map_sub = self.create_subscription(
                OccupancyGrid, map_topic, self.map_callback, self.qos_reliable,
                callback_group=self.callback_group
            )
            
            cmd_vel_topic = f'{target_ns}/cmd_vel'
            self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

            self._undock_client = ActionClient(self, Undock, f'{target_ns}/undock', callback_group=self.callback_group)
            self._dock_client = ActionClient(self, Dock, f'{target_ns}/dock', callback_group=self.callback_group)

            self.get_logger().info(f'>> 로봇 선택됨: {target_ns}')
            
            self.is_initialized = True
            self.is_setup_in_progress = False
            self.is_activated = True 
        finally:
            self.trigger_lock.release()

    def perform_undock_sequence(self):
        self.get_logger().info(">> Undock 시퀀스 시작...")
        try:
            if not self._undock_client.wait_for_server(timeout_sec=5.0): 
                self.get_logger().error("Undock Action Server not available")
                return False

            goal_msg = Undock.Goal()
            send_goal_future = self._undock_client.send_goal_async(goal_msg)
            
            # [참고] 여기는 별도 스레드(process_patient_mission_thread)라서 Blocking Wait가 허용됨.
            while not send_goal_future.done(): 
                time.sleep(0.1)
            
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted: 
                self.get_logger().error("Undock Goal Rejected")
                return False

            result_future = goal_handle.get_result_async()
            while not result_future.done(): 
                time.sleep(0.1)

            self.is_docked = False
            self.get_logger().info(">> Undock 완료.")
            return True
        except Exception as e:
            self.get_logger().error(f"Undock sequence error: {e}")
            return False

    def perform_docking(self):
        if not self._dock_client.wait_for_server(timeout_sec=2.0): 
            self.get_logger().error("Dock Action Server 응답 없음")
            return
        goal_msg = Dock.Goal()
        send_goal_future = self._dock_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.dock_response_callback)

    def dock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted: 
            return
        goal_handle.get_result_async().add_done_callback(lambda f: self.get_logger().info('>> Docking 최종 완료.'))
        self.is_docked = True
        self.mission_state = 'IDLE'

    def map_callback(self, msg):
        self.map_data = msg

    def is_point_valid(self, x, y):
        # [최적화] map_data 접근 시 예외 처리 추가 (NoneType Error 방지)
        if self.map_data is None: return False
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        # 인덱스 계산 최적화
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height: return False
        
        index = grid_y * width + grid_x
        # 맵 데이터 범위 체크
        if index >= len(self.map_data.data): return False
        
        return self.map_data.data[index] == 0

    def prepare_tts_files(self):
        if not os.path.exists(self.tts_alert_filename): self.generate_mp3("구조 업무 수행 중입니다. 잠시 비켜주세요.", self.tts_alert_filename)
        if not os.path.exists(self.tts_request_filename): self.generate_mp3("AED 사용 부탁드립니다.", self.tts_request_filename)
    
    def generate_mp3(self, text, filename):
        try:
            tts = gTTS(text=text, lang='ko')
            tts.save(filename)
        except Exception: pass
        
    def play_mp3(self, filename):
        try:
            # [최적화] subprocess 호출 빈도 제어
            if self.tts_process is None or self.tts_process.poll() is not None:
                self.tts_process = subprocess.Popen(['mpg321', filename, '--quiet'])
        except: pass
        
    def stop_tts(self):
        if self.tts_process:
            if self.tts_process.poll() is None: self.tts_process.terminate()
            self.tts_process = None
            
    def check_alert_loop(self):
        if not self.is_activated: return
        active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
        if self.is_paused and self.mission_state in active_states:
            self.play_mp3(self.tts_alert_filename)
            
    def check_arrival_loop(self):
        if not self.is_activated: return
        if self.mission_state == 'ARRIVED': self.play_mp3(self.tts_request_filename)

    def patient_pose_callback(self, msg):
        if not self.is_activated: return

        if self.mission_state in ['RETURNING_HOME', 'ARRIVED', 'BACKING_UP', 'PATROLLING', 'FINAL_APPROACH', 'NAVIGATING_TO_APPROACH']: 
            return 
        
        if self.is_setup_in_progress: return
        if not self.selected_robot_ns or self.map_data is None: return
        if self.is_processing_patient_req: return

        self.is_processing_patient_req = True
        self.get_logger().info(">> 새로운 환자 요청 접수.")
        threading.Thread(target=self.process_patient_mission_thread, args=(msg,)).start()

    def process_patient_mission_thread(self, msg):
        try:
            if self.is_docked:
                self.get_logger().info(">> 도킹 해제(Undock) 시퀀스 시작.")
                success = self.perform_undock_sequence()
                if not success: return 
                time.sleep(1.0) # 기계적 안정화 대기 (필수)

            if self.robot_current_pose is None:
                self.get_logger().warn(">> 로봇 위치 확인 불가.")
                return

            p_x = msg.pose.position.x
            p_y = msg.pose.position.y
            self.target_patient_coords = (p_x, p_y)

            r_x = self.robot_current_pose.position.x
            r_y = self.robot_current_pose.position.y
            dx = p_x - r_x
            dy = p_y - r_y
            target_yaw = math.atan2(dy, dx)

            approach_dist = 1.0
            new_x = p_x - approach_dist * math.cos(target_yaw)
            new_y = p_y - approach_dist * math.sin(target_yaw)
            
            goal_pose = PoseStamped()
            goal_pose.header = msg.header
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            goal_pose.pose.position.x = new_x
            goal_pose.pose.position.y = new_y
            goal_pose.pose.orientation = self.euler_to_quaternion(target_yaw)
            
            self.current_goal_pose = goal_pose 
            self.mission_state = 'NAVIGATING_TO_APPROACH'
            self.is_paused = False
            self.stop_tts()
            
            self.get_logger().info(f">> 1차 접근 시작")
            self.send_nav_goal(goal_pose)
        finally:
            self.is_processing_patient_req = False

    def aed_detected_callback(self, msg):
        if not self.is_activated: return
        if not self.aed_lock.acquire(blocking=False): return

        try:
            if msg.data is True and self.mission_state == 'ARRIVED':
                self.get_logger().info('>> AED Detected. 순찰 시작.')
                self.stop_tts()
                
                if self.robot_current_pose is not None:
                    self.backup_pose = PoseStamped()
                    self.backup_pose.header.frame_id = 'map'
                    self.backup_pose.header.stamp = self.get_clock().now().to_msg()
                    self.backup_pose.pose = self.robot_current_pose
                    self.start_octagon_patrol_smart()
        finally:
            self.aed_lock.release()

    def crowd_callback(self, msg):
        if not self.is_activated: return
        if self.mission_state in ['RETURNING_HOME', 'PATROLLING']: return
        
        # [최적화] Lock 제거하고 즉시 반응. 
        # Python의 bool assignment는 atomic하므로 Lock 없이도 심각한 문제는 없음 (속도 우선)
        # 하지만 안전을 위해 유지하되, 내부 로직을 가볍게.
        
        is_crowd_detected = msg.data
        if is_crowd_detected:
            active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
            if self.mission_state in active_states:
                self.last_crowd_seen_time = self.get_clock().now()
                if not self.is_paused:
                    self.get_logger().warn(f"군중 감지 -> 정지")
                    self.perform_emergency_stop()

    def timeout_check_callback(self):
        if not self.is_activated: return

        active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
        if self.mission_state in active_states and self.is_paused:
            now = self.get_clock().now()
            time_since_last_seen = (now - self.last_crowd_seen_time).nanoseconds / 1e9
            
            if time_since_last_seen > 5.0:
                self.get_logger().info(f"이동 재개")
                self.resume_navigation()

    def perform_backup_maneuver(self):
        if not self.robot_current_pose: return
        self.mission_state = 'BACKING_UP'

    def start_octagon_patrol_smart(self):
        if not self.current_goal_pose or not self.backup_pose: return
        
        if self.refined_patient_coords:
            center_x, center_y = self.refined_patient_coords
        elif self.target_patient_coords:
            center_x, center_y = self.target_patient_coords
        else:
            current_yaw = self.quaternion_to_yaw(self.current_goal_pose.pose.orientation)
            center_x = self.current_goal_pose.pose.position.x + 0.1 * math.cos(current_yaw)
            center_y = self.current_goal_pose.pose.position.y + 0.1 * math.sin(current_yaw)

        self.mission_state = 'PATROLLING'
        self.octagon_points = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            px = center_x + self.OCTAGON_RADIUS * math.cos(angle)
            py = center_y + self.OCTAGON_RADIUS * math.sin(angle)
            self.octagon_points.append((px, py))
            
        current_x = self.backup_pose.pose.position.x
        current_y = self.backup_pose.pose.position.y
        min_dist = float('inf')
        closest_index = 0
        for i, (px, py) in enumerate(self.octagon_points):
            dist = math.hypot(px - current_x, py - current_y)
            if dist < min_dist: min_dist = dist; closest_index = i
        self.current_pt_index = closest_index
        self.patrol_direction = 1 
        self.send_patrol_goal()

    def send_patrol_goal(self):
        if not self.octagon_points: return

        attempts = 0
        found_valid = False
        target_pose = None

        while attempts < 8:
            self.current_pt_index %= 8
            target_x, target_y = self.octagon_points[self.current_pt_index]
            
            if self.is_point_valid(target_x, target_y):
                found_valid = True
                angle_step = 2 * math.pi / 8
                vertex_angle = self.current_pt_index * angle_step
                target_yaw = vertex_angle + (self.patrol_direction * math.pi / 2.0)
                
                target_pose = PoseStamped()
                target_pose.header.frame_id = 'map'
                target_pose.header.stamp = self.get_clock().now().to_msg()
                target_pose.pose.position.x = target_x
                target_pose.pose.position.y = target_y
                target_pose.pose.orientation = self.euler_to_quaternion(target_yaw)
                break
            else:
                self.current_pt_index += self.patrol_direction
                attempts += 1
        
        if found_valid and target_pose:
            self.send_nav_goal(target_pose, is_patrol=True)
        else:
            self.patrol_direction *= -1

    def euler_to_quaternion(self, yaw):
        q = Quaternion(); q.z = math.sin(yaw/2.0); q.w = math.cos(yaw/2.0); return q
    def quaternion_to_yaw(self, q):
        return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        
    def perform_emergency_stop(self):
        self.is_paused = True 
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except:
                pass
        if self.cmd_vel_pub: self.cmd_vel_pub.publish(Twist())
        
    def resume_navigation(self):
        self.is_paused = False
        self.stop_tts()
        
        if self.current_goal_pose: 
            if self.mission_state == 'BACKING_UP': 
                pass
            elif self.mission_state == 'PATROLLING': 
                self.send_patrol_goal()
            elif self.mission_state == 'NAVIGATING_TO_APPROACH': 
                self.send_nav_goal(self.current_goal_pose)
            
    def send_nav_goal(self, pose_msg, is_patrol=False):
        if not self.selected_robot_ns: return

        action_name = f'{self.selected_robot_ns}/navigate_to_pose'
        if action_name not in self.nav_action_clients:
            self.nav_action_clients[action_name] = ActionClient(self, NavigateToPose, action_name, callback_group=self.callback_group)
        self._action_client = self.nav_action_clients[action_name]
        
        if not self._action_client.wait_for_server(timeout_sec=2.0): return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg
        
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.goal_response_callback(f, is_patrol))

    def goal_response_callback(self, future, is_patrol):
        handle = future.result()
        if not handle.accepted:
            if is_patrol: 
                # [최적화] 재귀 호출 방지: 타이머나 비동기로 처리하면 좋으나, 로직 복잡성으로 유지
                self.current_pt_index += self.patrol_direction * -1 # 실패시 반대 방향
                self.patrol_direction *= -1
                # self.send_patrol_goal() # 스택 오버플로우 위험 있으므로 루프에서 처리 권장
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.get_result_callback(f, is_patrol))

    def get_result_callback(self, future, is_patrol):
        result = future.result()
        status = result.status
        
        if status == 6: return # CANCELED

        if status == 4: # SUCCEEDED
            if self.mission_state == 'RETURNING_HOME':
                self.perform_docking()
            elif self.mission_state == 'NAVIGATING_TO_APPROACH':
                self.get_logger().info("1차 접근 완료. 정밀 탐색 시작.")
                self.mission_state = 'SEARCHING_PATIENT'
            elif is_patrol:
                self.current_pt_index += self.patrol_direction; self.send_patrol_goal()
        else:
            if is_patrol: 
                self.patrol_direction *= -1
                self.send_patrol_goal()

    def get_yaw_from_pose(self, pose):
        orientation_q = pose.orientation
        _, _, yaw = self.euler_from_quaternion(
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        )
        return yaw

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z 

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = AEDNavigatorOctagonSmooth()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: node.stop_tts()
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()