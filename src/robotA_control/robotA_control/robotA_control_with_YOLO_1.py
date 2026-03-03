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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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

        self.qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = None 
        self.cmd_vel_pub = None 
        
        self._undock_client = None
        self._dock_client = None

        # --- [Subscribers] ---
        self.trigger_sub = self.create_subscription(
            Bool, '/robot_role', self.trigger_callback, 10, 
            callback_group=self.callback_group)
        self.pose_sub = self.create_subscription(
            PoseStamped, 'patient_pose', self.patient_pose_callback, 10, 
            callback_group=self.callback_group)
        
        self.crowd_sub = self.create_subscription(
            Bool, '/robotA/crowd_detected', self.crowd_callback, 10, 
            callback_group=self.callback_group)
            
        self.aed_detected_sub = self.create_subscription(
            Bool, '/robotA/aed_detected', self.aed_detected_callback, 10, 
            callback_group=self.callback_group)
            
        self.stop_sub = self.create_subscription(
            Bool, '/robot_stop', self.robot_stop_callback, 10,
            callback_group=self.callback_group)

        self.patient_dist_sub = self.create_subscription(
            Float32, '/robotA/patient/distance', self.patient_dist_callback, 10,
            callback_group=self.callback_group)

        self.robot_current_pose = None
        self.target_patient_coords = None
        self.refined_patient_coords = None

        self.create_subscription(PoseWithCovarianceStamped, '/robot3/amcl_pose', 
                                 lambda m: self.amcl_pose_callback(m, '/robot3'), 
                                 self.qos_profile, callback_group=self.callback_group)
        self.create_subscription(PoseWithCovarianceStamped, '/robot5/amcl_pose', 
                                 lambda m: self.amcl_pose_callback(m, '/robot5'), 
                                 self.qos_profile, callback_group=self.callback_group)

        # --- [Variables] ---
        self._action_client = None
        self._goal_handle = None        
        self.selected_robot_ns = None   
        self.current_goal_pose = None
        
        # [수정] 수동 백업을 위한 시작 위치 저장 변수
        self.backup_start_pose = None
        self.backup_pose = None # 완료 후 위치

        self.map_data = None  
        
        self.mission_state = 'IDLE'   
        self.is_paused = False
        
        self.last_crowd_seen_time = self.get_clock().now()
        
        self.tts_process = None 
        self.is_initialized = False 
        self.is_docked = True       

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
        
        # 타이머
        self.create_timer(0.1, self.timeout_check_callback, callback_group=self.callback_group)
        self.create_timer(0.1, self.check_alert_loop, callback_group=self.callback_group)   
        self.create_timer(2.0, self.check_arrival_loop, callback_group=self.callback_group) 
        self.create_timer(0.1, self.control_loop_callback, callback_group=self.callback_group)

        self.get_logger().info('=== AED Octagon Smooth Patrol Ready (Modified: Skip Backup) ===')

    def amcl_pose_callback(self, msg, ns):
        if self.selected_robot_ns == ns:
            self.robot_current_pose = msg.pose.pose

    def patient_dist_callback(self, msg):
        self.latest_patient_dist = msg.data
        self.last_patient_dist_time = self.get_clock().now()

        if self.robot_current_pose is not None and msg.data > 0.0:
            q = self.robot_current_pose.orientation
            current_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            calc_x = self.robot_current_pose.position.x + msg.data * math.cos(current_yaw)
            calc_y = self.robot_current_pose.position.y + msg.data * math.sin(current_yaw)
            self.refined_patient_coords = (calc_x, calc_y)

    def control_loop_callback(self):
        # Pause 상태 처리
        if self.is_paused:
            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(Twist()) # 정지
            return

        # BACKING_UP 상태는 여기서 직접 제어하기 위해 return 목록에서 제외함
        if self.mission_state in ['IDLE', 'NAVIGATING_TO_APPROACH', 'RETURNING_HOME', 'PATROLLING', 'ARRIVED']:
            return

        twist_msg = Twist()
        
        # ---------------------------------------------------------
        # [참고] AED 감지 시 이 로직(BACKING_UP)은 이제 건너뜁니다.
        # ---------------------------------------------------------
        if self.mission_state == 'BACKING_UP':
            if self.robot_current_pose is None:
                self.get_logger().warn("백업 불가: 로봇 위치 정보 없음")
                return

            # 1. 초기화 단계
            if not hasattr(self, 'backup_phase') or self.backup_phase is None:
                self.backup_phase = 'TURN_180_1'
                self.backup_start_pose = self.robot_current_pose
                current_yaw = self.get_yaw_from_pose(self.robot_current_pose)
                self.target_yaw_1 = self.normalize_angle(current_yaw + math.pi)
                self.target_yaw_2 = current_yaw 
                self.get_logger().info("백업 시퀀스 시작: 1단계 180도 회전 진입")

            # 2. 단계별 실행 로직
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
                    self.get_logger().info(f"전진 완료 ({dist_moved:.2f}m). 원위치 회전 시작.")

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
        # ---------------------------------------------------------
        
        time_diff = (self.get_clock().now() - self.last_patient_dist_time).nanoseconds / 1e9
        is_data_fresh = time_diff < 1.0

        if self.mission_state == 'SEARCHING_PATIENT':
            if is_data_fresh:
                self.get_logger().info(f"환자 발견! (거리: {self.latest_patient_dist:.2f}m). 접근 시작.")
                self.mission_state = 'FINAL_APPROACH'
                self.cmd_vel_pub.publish(twist_msg)
            else:
                twist_msg.angular.z = 0.3 
                self.cmd_vel_pub.publish(twist_msg)

        elif self.mission_state == 'FINAL_APPROACH':
            if not is_data_fresh:
                self.get_logger().warn("환자 신호 상실! 다시 탐색 모드로 전환.")
                self.mission_state = 'SEARCHING_PATIENT'
                self.cmd_vel_pub.publish(twist_msg)
            else:
                if self.latest_patient_dist > 1.0 or self.latest_patient_dist is None or self.latest_patient_dist == 0.0:
                    twist_msg.linear.x = 0.05
                    self.cmd_vel_pub.publish(twist_msg)
                else:
                    self.get_logger().info(f"최종 도착 완료! (거리: {self.latest_patient_dist:.2f}m)")
                    self.cmd_vel_pub.publish(twist_msg)
                    self.mission_state = 'ARRIVED'
                    self.play_mp3(self.tts_request_filename)

    def robot_stop_callback(self, msg):
        if msg.data is True:
            self.get_logger().warn("!!! 비상 정지 명령 수신 !!! 모든 작업 중단 및 복귀.")
            if self._goal_handle is not None:
                try:
                    self._goal_handle.cancel_goal_async()
                except:
                    pass
            self.stop_tts()
            self.is_paused = False 
            self.mission_state = 'RETURNING_HOME'

            if self.cmd_vel_pub:
                self.cmd_vel_pub.publish(Twist())

            home_pose = PoseStamped()
            home_pose.header.frame_id = 'map'
            home_pose.header.stamp = self.get_clock().now().to_msg()
            
            if self.selected_robot_ns == '/robot3':
                home_pose.pose.position.x = 0.5
                home_pose.pose.position.y = 0.5
                home_pose.pose.orientation.w = 1.0 
            elif self.selected_robot_ns == '/robot5':
                home_pose.pose.position.x = -1.35
                home_pose.pose.position.y = -4.58
                home_pose.pose.orientation.w = 0.707
                home_pose.pose.orientation.z = 0.707
            else:
                return
            self.send_nav_goal(home_pose)

    def trigger_callback(self, msg):
        target_ns = '/robot3' if msg.data else '/robot5'
        if self.selected_robot_ns == target_ns: return

        self.selected_robot_ns = target_ns
        self.robot_current_pose = None
        
        if self.map_sub is not None:
            self.destroy_subscription(self.map_sub)
            self.map_data = None 
        if self.cmd_vel_pub is not None:
            self.destroy_publisher(self.cmd_vel_pub)

        map_topic = f'{target_ns}/map'
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self.map_callback, self.qos_profile,
            callback_group=self.callback_group
        )
        
        cmd_vel_topic = f'{target_ns}/cmd_vel'
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self._undock_client = ActionClient(self, Undock, f'{target_ns}/undock', callback_group=self.callback_group)
        self._dock_client = ActionClient(self, Dock, f'{target_ns}/dock', callback_group=self.callback_group)

        self.get_logger().info(f'>> 로봇 선택됨: {target_ns}')

        if not self.is_initialized and self.is_docked:
            threading.Thread(target=self.perform_startup_sequence).start()
            self.is_initialized = True

    def perform_startup_sequence(self):
        self.get_logger().info(">> 초기화 시퀀스 시작: Undock Action 요청")
        if not self._undock_client.wait_for_server(timeout_sec=5.0): return

        goal_msg = Undock.Goal()
        send_goal_future = self._undock_client.send_goal_async(goal_msg)
        while not send_goal_future.done(): time.sleep(0.1)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted: return

        result_future = goal_handle.get_result_async()
        while not result_future.done(): time.sleep(0.1)

        move_msg = Twist()
        move_msg.linear.x = 0.05
        for _ in range(75):
            self.cmd_vel_pub.publish(move_msg)
            time.sleep(0.1)
        move_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(move_msg)
        self.is_docked = False

    def perform_docking(self):
        if not self._dock_client.wait_for_server(timeout_sec=5.0): return
        goal_msg = Dock.Goal()
        send_goal_future = self._dock_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.dock_response_callback)

    def dock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted: return
        goal_handle.get_result_async().add_done_callback(lambda f: self.get_logger().info('>> Docking 완료.'))
        self.is_docked = True
        self.mission_state = 'IDLE'

    def map_callback(self, msg):
        self.map_data = msg

    def is_point_valid(self, x, y):
        if self.map_data is None: return False
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        height = self.map_data.info.height
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height: return False
        index = grid_y * width + grid_x
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
            if self.tts_process is None or self.tts_process.poll() is not None:
                self.tts_process = subprocess.Popen(['mpg321', filename, '--quiet'])
        except: pass
    def stop_tts(self):
        if self.tts_process:
            if self.tts_process.poll() is None: self.tts_process.terminate()
            self.tts_process = None
            
    def check_alert_loop(self):
        active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
        if self.is_paused and self.mission_state in active_states:
            self.play_mp3(self.tts_alert_filename)
            
    def check_arrival_loop(self):
        if self.mission_state == 'ARRIVED': self.play_mp3(self.tts_request_filename)

    def patient_pose_callback(self, msg):
        if self.mission_state in ['RETURNING_HOME', 'ARRIVED', 'BACKING_UP', 'PATROLLING', 'FINAL_APPROACH']: return 
        if not self.selected_robot_ns or self.robot_current_pose is None or self.map_data is None: return

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
        
        msg.pose.position.x = new_x
        msg.pose.position.y = new_y
        msg.pose.orientation = self.euler_to_quaternion(target_yaw)
        msg.header.stamp = self.get_clock().now().to_msg()
        
        self.current_goal_pose = msg 
        self.mission_state = 'NAVIGATING_TO_APPROACH'
        self.is_paused = False
        self.stop_tts()
        
        self.get_logger().info(f">> 1차 접근 시작: 목표({new_x:.2f}, {new_y:.2f})")
        self.send_nav_goal(msg)

    def aed_detected_callback(self, msg):
        # [수정된 부분] AED 감지 시 후진(Backup) 없이 바로 순찰(Patrol) 시작
        if msg.data is True and self.mission_state == 'ARRIVED':
            self.get_logger().info('>> 전달 확인(AED Detected). 후진 생략 -> 즉시 순찰 시작.')
            self.stop_tts()
            
            # 현재 위치를 기준으로 순찰을 시작하기 위해 변수 설정
            if self.robot_current_pose is not None:
                self.backup_pose = PoseStamped()
                self.backup_pose.header.frame_id = 'map'
                self.backup_pose.header.stamp = self.get_clock().now().to_msg()
                self.backup_pose.pose = self.robot_current_pose
                
                # 바로 순찰 로직 호출
                self.start_octagon_patrol_smart()
            else:
                self.get_logger().warn("현재 로봇 위치를 알 수 없어 순찰을 시작할 수 없습니다.")

    def crowd_callback(self, msg):
        if self.mission_state == 'RETURNING_HOME': return 
        
        is_crowd_detected = msg.data
        active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
        
        if self.mission_state in active_states:
            if is_crowd_detected:
                self.last_crowd_seen_time = self.get_clock().now()
                if not self.is_paused:
                    self.get_logger().warn(f"[{self.mission_state}] 최초 군중 감지 -> 비상 정지 실행.")
                    self.perform_emergency_stop()

    def timeout_check_callback(self):
        active_states = ['NAVIGATING_TO_APPROACH', 'BACKING_UP', 'PATROLLING', 'SEARCHING_PATIENT', 'FINAL_APPROACH']
        
        if self.mission_state in active_states and self.is_paused:
            now = self.get_clock().now()
            time_since_last_seen = (now - self.last_crowd_seen_time).nanoseconds / 1e9
            
            if time_since_last_seen > 5.0:
                self.get_logger().info(f"군중 미감지 {time_since_last_seen:.1f}초 경과. 이동 재개.")
                self.resume_navigation()

    def perform_backup_maneuver(self):
        # [참고] 이제 이 함수는 AED 콜백에서 호출되지 않습니다.
        if not self.robot_current_pose:
            self.get_logger().warn("백업 실패: 로봇 위치 모름")
            return
            
        self.get_logger().info(">> 수동 후진(Manual Backup) 모드 진입")
        self.backup_start_pose = self.robot_current_pose 
        self.mission_state = 'BACKING_UP'

    def start_octagon_patrol_smart(self):
        # backup_pose는 위 콜백에서 수동으로 설정됨
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

    def perform_wall_retreat(self):
        if self._goal_handle: self._goal_handle.cancel_goal_async()
        self.current_pt_index -= self.patrol_direction
        self.patrol_direction *= -1
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
    def reverse_patrol_direction(self):
        if self._goal_handle: self._goal_handle.cancel_goal_async()
        self.patrol_direction *= -1
        self.current_pt_index += (self.patrol_direction * 2)
        self.send_patrol_goal()
        
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
        self.get_logger().info("이동 재개")
        
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
            if is_patrol: self.reverse_patrol_direction()
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.get_result_callback(f, is_patrol))

    def get_result_callback(self, future, is_patrol):
        result = future.result()
        status = result.status
        
        if status == 6: # CANCELED
            return

        if status == 4: # SUCCEEDED
            if self.mission_state == 'RETURNING_HOME':
                self.perform_docking()
            elif self.mission_state == 'NAVIGATING_TO_APPROACH':
                self.get_logger().info("1차 접근 완료. 환자 정밀 탐색 시작.")
                self.mission_state = 'SEARCHING_PATIENT'
            elif not is_patrol and self.mission_state == 'MOVING_TO_PATIENT': 
                pass
            elif is_patrol:
                self.current_pt_index += self.patrol_direction; self.send_patrol_goal()
        elif status == 5: pass
        else:
            if is_patrol: self.reverse_patrol_direction()

    def get_yaw_from_pose(self, pose):
        """Pose 메시지에서 Yaw 각도(라디안) 추출"""
        orientation_q = pose.orientation
        _, _, yaw = self.euler_from_quaternion(
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        )
        return yaw

    def euler_from_quaternion(self, x, y, z, w):
        """쿼터니언 -> 오일러 변환 (roll, pitch, yaw)"""
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
     
        return roll_x, pitch_y, yaw_z # 라디안 단위

    def normalize_angle(self, angle):
        """각도를 -pi ~ pi 사이로 정규화"""
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