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

# [추가] irobot_create_msgs (TurtleBot4/Create3 표준 액션)
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
        
        # [수정] Undock/Dock용 Action Client 변수
        self._undock_client = None
        self._dock_client = None

        # --- [Subscribers] ---
        self.trigger_sub = self.create_subscription(
            Bool, '/robot_role', self.trigger_callback, 10, 
            callback_group=self.callback_group)
        self.pose_sub = self.create_subscription(
            PoseStamped, 'patient_pose', self.patient_pose_callback, 10, 
            callback_group=self.callback_group)
        self.distance_sub = self.create_subscription(
            Float32, '/people_distance', self.distance_callback, 10, 
            callback_group=self.callback_group)
        self.transfer_sub = self.create_subscription(
            Bool, '/aed_transfer', self.transfer_callback, 10, 
            callback_group=self.callback_group)
        self.stop_sub = self.create_subscription(
            Bool, '/robot_stop', self.robot_stop_callback, 10,
            callback_group=self.callback_group)

        # --- [AMCL Pose] ---
        self.create_subscription(PoseWithCovarianceStamped, '/robot3/amcl_pose', lambda m: None, self.qos_profile, callback_group=self.callback_group)
        self.create_subscription(PoseWithCovarianceStamped, '/robot5/amcl_pose', lambda m: None, self.qos_profile, callback_group=self.callback_group)

        # --- [Variables] ---
        self._action_client = None
        self._goal_handle = None        
        self.selected_robot_ns = None   
        self.current_goal_pose = None
        self.backup_pose = None
        self.map_data = None  
        
        self.mission_state = 'IDLE'   
        self.is_paused = False          
        self.last_dist_time = self.get_clock().now() 
        self.tts_process = None 
        
        self.is_initialized = False 
        self.is_docked = True       

        self.octagon_points = []
        self.current_pt_index = 0
        self.patrol_direction = 1     
        self.OCTAGON_RADIUS = 0.6

        self.tts_alert_filename = "emergency_alert.mp3"
        self.tts_request_filename = "aed_request.mp3"
        self.prepare_tts_files()
        
        self.create_timer(1.0, self.timeout_check_callback, callback_group=self.callback_group)
        self.create_timer(0.5, self.check_alert_loop, callback_group=self.callback_group)   
        self.create_timer(2.0, self.check_arrival_loop, callback_group=self.callback_group) 

        self.get_logger().info('=== AED Octagon Smooth Patrol Ready (Native Action) ===')
        self.get_logger().info('Waiting for /robot_role to trigger sequence...')

    def robot_stop_callback(self, msg):
        if msg.data is True:
            self.get_logger().warn("!!! 비상 정지 명령 수신 !!! 모든 작업 중단 및 복귀.")
            if self._goal_handle:
                self._goal_handle.cancel_goal_async()
            self.stop_tts()
            self.is_paused = False 
            self.mission_state = 'RETURNING_HOME'

            home_pose = PoseStamped()
            home_pose.header.frame_id = 'map'
            home_pose.header.stamp = self.get_clock().now().to_msg()
            home_pose.pose.orientation.w = 1.0 

            if self.selected_robot_ns == '/robot3':
                home_pose.pose.position.x = 0.2
                home_pose.pose.position.y = 0.0
                self.get_logger().info(">> Robot3 홈(0.2, 0.0)으로 이동합니다.")
            elif self.selected_robot_ns == '/robot5':
                home_pose.pose.position.x = -1.35
                home_pose.pose.position.y = -4.58
                self.get_logger().info(">> Robot5 홈(-1.35, -4.58)으로 이동합니다.")
            else:
                self.get_logger().error("로봇이 선택되지 않아 홈으로 이동할 수 없습니다.")
                return

            self.send_nav_goal(home_pose)

    def trigger_callback(self, msg):
        target_ns = '/robot3' if msg.data else '/robot5'
        
        if self.selected_robot_ns == target_ns: return

        self.selected_robot_ns = target_ns
        
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

        # [수정] Undock/Dock 액션 클라이언트 생성 (네임스페이스 적용)
        self._undock_client = ActionClient(self, Undock, f'{target_ns}/undock', callback_group=self.callback_group)
        self._dock_client = ActionClient(self, Dock, f'{target_ns}/dock', callback_group=self.callback_group)

        self.get_logger().info(f'>> 로봇 선택됨: {target_ns}')

        if not self.is_initialized and self.is_docked:
            threading.Thread(target=self.perform_startup_sequence).start()
            self.is_initialized = True

    def perform_startup_sequence(self):
        self.get_logger().info(">> 초기화 시퀀스 시작: Undock Action 요청")

        # 1. Undock Action 호출 (Native)
        if not self._undock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Undock Action Server not available!")
            return

        goal_msg = Undock.Goal()
        # 동기 호출(wait_for_result) 대신 Future 사용 (스레드 내부이므로 wait 가능)
        send_goal_future = self._undock_client.send_goal_async(goal_msg)
        
        # Future가 완료될 때까지 대기 (간단한 busy wait)
        while not send_goal_future.done():
            time.sleep(0.1)
        
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Undock goal rejected.')
            return

        self.get_logger().info('Undock goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        
        while not result_future.done():
            time.sleep(0.1)

        self.get_logger().info("   [Action] Undocking complete.") 
        
        # 2. 20cm 전진
        move_msg = Twist()
        move_msg.linear.x = 0.20
        move_msg.angular.z = 0.0

        self.get_logger().info("   [Move] 20cm 전진 중...")
        for _ in range(15): # 1.5초 (약간 넉넉하게)
            self.cmd_vel_pub.publish(move_msg)
            time.sleep(0.1)

        move_msg.linear.x = 0.0
        self.cmd_vel_pub.publish(move_msg)
        
        self.is_docked = False
        self.get_logger().info(">> 초기화 완료. 명령 대기 중.")

    def perform_docking(self):
        self.get_logger().info(">> 홈 도착 완료. Docking Action 시작.")
        
        if not self._dock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Dock Action Server not available!")
            return

        goal_msg = Dock.Goal()
        send_goal_future = self._dock_client.send_goal_async(goal_msg)
        
        # 스레드가 아니라 콜백 내부에서 불릴 수 있으므로, 
        # Future에 콜백을 거는 방식이 안전하지만, 
        # 여기서는 복귀 후 마지막 동작이므로 비동기로 던져놓음.
        send_goal_future.add_done_callback(self.dock_response_callback)

    def dock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Dock goal rejected.')
            return
        self.get_logger().info('Dock goal accepted.')
        goal_handle.get_result_async().add_done_callback(lambda f: self.get_logger().info('>> Docking 완료. 시스템 대기.'))
        self.is_docked = True
        self.mission_state = 'IDLE'

    # ---------------- 기존 콜백 및 함수들 (그대로 유지) ----------------
    def map_callback(self, msg):
        self.map_data = msg
        self.get_logger().info(f'>> 지도 수신 완료! ({msg.info.width}x{msg.info.height})', throttle_duration_sec=10)

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
        alert_text = "구조 업무 수행 중입니다. 잠시 비켜주세요."
        request_text = "AED 사용 부탁드립니다." 
        if not os.path.exists(self.tts_alert_filename): self.generate_mp3(alert_text, self.tts_alert_filename)
        if os.path.exists(self.tts_request_filename): os.remove(self.tts_request_filename)
        self.generate_mp3(request_text, self.tts_request_filename)
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
        if self.is_paused and self.mission_state == 'MOVING_TO_PATIENT': self.play_mp3(self.tts_alert_filename)
    def check_arrival_loop(self):
        if self.mission_state == 'ARRIVED': self.play_mp3(self.tts_request_filename)

    def patient_pose_callback(self, msg):
        if self.mission_state == 'RETURNING_HOME': return 
        if not self.selected_robot_ns: 
            self.get_logger().warn('로봇 선택 필요')
            return
        if self.map_data is None:
            self.get_logger().warn('지도 수신 대기 중...')
            # 메인 스레드 블로킹 방지를 위해 타이머 등으로 재시도하는 것이 좋으나 기존 로직 유지
            # 주의: 여기서는 sleep을 쓰면 콜백을 막을 수 있음. 
            if self.map_data is None: return 

        self.get_logger().info('>> 환자 위치 수신. 네비게이션 시작 준비 중...') 
        target_yaw = self.quaternion_to_yaw(msg.pose.orientation)
        approach_dist = 0.1
        new_x = msg.pose.position.x - approach_dist * math.cos(target_yaw)
        new_y = msg.pose.position.y - approach_dist * math.sin(target_yaw)
        msg.pose.position.x = new_x
        msg.pose.position.y = new_y
        msg.header.stamp = self.get_clock().now().to_msg()
        self.current_goal_pose = msg 
        self.mission_state = 'MOVING_TO_PATIENT'
        self.is_paused = False
        self.stop_tts()
        self.send_nav_goal(msg)

    def transfer_callback(self, msg):
        if msg.data is True and self.mission_state == 'ARRIVED':
            self.get_logger().info('>> 전달 확인. 후진 시작.')
            self.stop_tts()
            self.perform_backup_maneuver()

    def distance_callback(self, msg):
        if self.mission_state == 'RETURNING_HOME': return 
        self.last_dist_time = self.get_clock().now()
        distance = msg.data
        if self.mission_state == 'MOVING_TO_PATIENT':
            if distance < 1.0 and not self.is_paused: self.perform_emergency_stop()
            elif distance >= 1.0 and self.is_paused: self.resume_navigation()
        elif self.mission_state == 'PATROLLING':
            if distance < 0.4: self.perform_wall_retreat()

    def timeout_check_callback(self):
        if self.mission_state == 'MOVING_TO_PATIENT' and self.is_paused:
            if (self.get_clock().now() - self.last_dist_time).nanoseconds / 1e9 > 2.0:
                self.resume_navigation()

    def perform_backup_maneuver(self):
        if not self.current_goal_pose: return
        p_x = self.current_goal_pose.pose.position.x
        p_y = self.current_goal_pose.pose.position.y
        p_q = self.current_goal_pose.pose.orientation
        patient_yaw = self.quaternion_to_yaw(p_q)
        back_x = p_x - self.OCTAGON_RADIUS * math.cos(patient_yaw)
        back_y = p_y - self.OCTAGON_RADIUS * math.sin(patient_yaw)
        if not self.is_point_valid(back_x, back_y):
            self.backup_pose = self.current_goal_pose 
            self.start_octagon_patrol_smart()
            return
        self.backup_pose = PoseStamped()
        self.backup_pose.header.frame_id = 'map'
        self.backup_pose.header.stamp = self.get_clock().now().to_msg()
        self.backup_pose.pose.position.x = back_x
        self.backup_pose.pose.position.y = back_y
        self.backup_pose.pose.orientation = p_q 
        self.mission_state = 'BACKING_UP'
        self.send_nav_goal(self.backup_pose)

    def start_octagon_patrol_smart(self):
        if not self.current_goal_pose or not self.backup_pose: return
        self.mission_state = 'PATROLLING'
        current_yaw = self.quaternion_to_yaw(self.current_goal_pose.pose.orientation)
        center_x = self.current_goal_pose.pose.position.x + 0.1 * math.cos(current_yaw)
        center_y = self.current_goal_pose.pose.position.y + 0.1 * math.sin(current_yaw)
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
        self.current_pt_index %= 8
        target_x, target_y = self.octagon_points[self.current_pt_index]
        if not self.is_point_valid(target_x, target_y):
            self.patrol_direction *= -1
            self.current_pt_index += (self.patrol_direction * 2)
            self.send_patrol_goal()
            return
        angle_step = 2 * math.pi / 8
        vertex_angle = self.current_pt_index * angle_step
        target_yaw = vertex_angle + (self.patrol_direction * math.pi / 2.0)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = target_x
        pose.pose.position.y = target_y
        pose.pose.orientation = self.euler_to_quaternion(target_yaw)
        self.send_nav_goal(pose, is_patrol=True)

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
        if self._goal_handle: self._goal_handle.cancel_goal_async()
        self.play_mp3(self.tts_alert_filename)
    def resume_navigation(self):
        self.is_paused = False
        self.stop_tts()
        if self.current_goal_pose: 
            if self.mission_state == 'BACKING_UP': self.send_nav_goal(self.backup_pose)
            elif self.mission_state == 'PATROLLING': self.send_patrol_goal()
            else: self.send_nav_goal(self.current_goal_pose)
            
    def send_nav_goal(self, pose_msg, is_patrol=False):
        action_name = f'{self.selected_robot_ns}/navigate_to_pose'
        self._action_client = ActionClient(self, NavigateToPose, action_name, callback_group=self.callback_group)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose_msg
        if not self._action_client.wait_for_server(timeout_sec=2.0): return
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
        status = future.result().status
        if status == 4: # SUCCEEDED
            if self.mission_state == 'RETURNING_HOME':
                self.perform_docking()
            elif self.mission_state == 'BACKING_UP':
                self.start_octagon_patrol_smart()
            elif not is_patrol and self.mission_state == 'MOVING_TO_PATIENT': 
                self.mission_state = 'ARRIVED'; self.play_mp3(self.tts_request_filename)
            elif is_patrol:
                self.current_pt_index += self.patrol_direction; self.send_patrol_goal()
        elif status == 5: pass
        else:
            if is_patrol: self.reverse_patrol_direction()

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