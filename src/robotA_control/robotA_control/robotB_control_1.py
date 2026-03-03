#!/usr/bin/env python3
"""
robot_b.py (이벤트 기반 리팩토링 - 성능 최적화 버전)
병목 제거: Blocking TF Lookup 제거, 상태 체크 주기 단축
"""

import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from tf_transformations import quaternion_from_euler, euler_from_quaternion

import tf2_ros
import tf2_msgs.msg
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration


# ==========================================
# 설정 파라미터
# ==========================================
OFFSET_DISTANCE = 0.6
RESCUE_POSITION = (2.31, -1.36, 270.0)
DOCK_APPROACH_POSITION_ROBOT3 = (-0.42, -0.07, 90.0)
DOCK_APPROACH_POSITION_ROBOT5 = (-1.32, -3.82, 90.0)


class RescueRobotEventBased(Node):
    def __init__(self):
        super().__init__('rescue_b')
        
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        # Navigator 초기화
        self.robot3_dock_nav = TurtleBot4Navigator(namespace='robot3')
        self.robot3_nav = BasicNavigator(namespace='robot3')
        
        self.robot5_dock_nav = TurtleBot4Navigator(namespace='robot5')
        self.robot5_nav = BasicNavigator(namespace='robot5')
        
        # TF Buffer
        self.tf_buffer_robot3 = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_buffer_robot5 = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        
        # TF 구독
        static_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_ALL,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        
        self.create_subscription(tf2_msgs.msg.TFMessage, '/robot3/tf', self.tf3_callback, 10)
        self.create_subscription(tf2_msgs.msg.TFMessage, '/robot3/tf_static', self.tf3_static_callback, static_qos)
        self.create_subscription(tf2_msgs.msg.TFMessage, '/robot5/tf', self.tf5_callback, 10)
        self.create_subscription(tf2_msgs.msg.TFMessage, '/robot5/tf_static', self.tf5_static_callback, static_qos)
        
        # 상태 변수
        self.target_coordinate = None
        self.robot_selected = None
        self.is_moving = False
        self.is_paused = False
        self.current_goal_pose = None
        self.crowd_detected = False
        
        # 이벤트 플래그
        self.mission_started = False
        self.waiting_for_rescue = False
        self.rescue_detected = False
        self.waiting_for_stop = False
        
        # 상태 퍼블리셔
        self.progress_pub = self.create_publisher(Int32, '/robotB/progress', 10)

        # 🔊 Beep 퍼블리셔
        self.beep_pub_robot3 = self.create_publisher(AudioNoteVector, '/robot3/cmd_audio', 10)
        self.beep_pub_robot5 = self.create_publisher(AudioNoteVector, '/robot5/cmd_audio', 10)
        self.beep_timer = None
        
        # 구독자들
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/robot_role', self.robot_select_callback, qos)
        self.create_subscription(PoseStamped, '/patient_pose', self.goal_callback, qos)
        self.create_subscription(Bool, '/robotB/responder_done', self.responder_done_callback, qos)
        self.create_subscription(Bool, '/robot_stop', self.stop_callback, qos)
        self.create_subscription(Bool, '/robotB/crowd_detected', self.crowd_callback, 10)
        
        # [최적화] 상태 체크 주기 단축 (0.5s -> 0.1s)
        # 반응성 향상 (Ping 개선)
        self.create_timer(0.1, self.check_navigation_status)
        
        self.get_logger().info("✅ RescueRobot 초기화 완료 (성능 최적화됨)")
        
    # ==========================================
    # TF 콜백들
    # ==========================================
    def tf3_callback(self, msg):
        for t in msg.transforms:
            self.tf_buffer_robot3.set_transform(t, 'default_authority')
    
    def tf3_static_callback(self, msg):
        for t in msg.transforms:
            self.tf_buffer_robot3.set_transform(t, 'default_authority')
    
    def tf5_callback(self, msg):
        for t in msg.transforms:
            self.tf_buffer_robot5.set_transform(t, 'default_authority')
    
    def tf5_static_callback(self, msg):
        for t in msg.transforms:
            self.tf_buffer_robot5.set_transform(t, 'default_authority')
    
    # ==========================================
    # 이벤트 콜백들
    # ==========================================
    def robot_select_callback(self, msg: Bool):
        if self.mission_started:
            return
        if msg.data:
            self.robot_selected = 'robot5'
            self.get_logger().info(f"🤖 로봇 선택: {self.robot_selected}")
            self.try_start_mission()

    def goal_callback(self, msg: PoseStamped):
        if self.mission_started:
            return
        self.target_coordinate = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"📍 목표 좌표: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        self.try_start_mission()
    
    def responder_done_callback(self, msg: Bool):
        if msg.data and self.waiting_for_rescue:
            self.get_logger().info("🚑 구급대원 발견! 2차 이동 시작")
            self.rescue_detected = True
            self.waiting_for_rescue = False
            self._rescue_timer = self.create_timer(1.0, self.move_to_target_second)
    
    def stop_callback(self, msg: Bool):
        if not self.mission_started or not msg.data:
            return

        self.get_logger().warn("🛑 긴급 중단 명령 수신! 즉시 도킹 스테이션으로 복귀합니다.")
        
        nav, _, _ = self.get_current_navigator()
        if self.is_moving:
            nav.cancelTask()
            self.is_moving = False
        
        self.waiting_for_rescue = False
        self.waiting_for_stop = False
        self._nav_complete_callback = None
        self.cancel_all_timers()
        self.stop_beep()

        self.create_timer(0.5, self.move_to_dock)

    def cancel_all_timers(self):
        timer_attrs = [
            '_safety_timer',
            '_first_arrival_timer', 
            '_rescue_timer',
            '_second_arrival_timer',
            '_undock_timer',
            '_dock_timer',
            '_tf_retry_timer' # [추가] 재시도 타이머
        ]
        
        for attr in timer_attrs:
            if hasattr(self, attr):
                timer = getattr(self, attr)
                if timer is not None:
                    timer.cancel()
                    self.get_logger().info(f"⏹️ {attr} 취소됨")
    
    def crowd_callback(self, msg: Bool):
        self.crowd_detected = msg.data
        if not (self.is_moving or self.is_paused):
            return
        
        nav, _, _ = self.get_current_navigator()
        
        if self.crowd_detected and not self.is_paused:
            self.is_paused = True
            self.is_moving = False
            nav.cancelTask()
            self.get_logger().warn("🛑 군중 감지! 정지")
        
        elif not self.crowd_detected and self.is_paused:
            self.is_paused = False
            self.is_moving = True
            if self.current_goal_pose is not None:
                nav.goToPose(self.current_goal_pose)
                self.get_logger().info("✅ 군중 해소! 재개")

    # ==========================================
    # Beep 관련 함수들
    # ==========================================
    def create_beep_message(self):          
        msg = AudioNoteVector()
        msg.append = False
        note = AudioNote()
        note.frequency = 880
        note.max_runtime = Duration(sec=0, nanosec=100000000)
        msg.notes = [note]
        return msg
    
    def start_beep(self):
        if self.beep_timer is None:
            self.beep_timer = self.create_timer(0.5, self.beep_callback)
            self.get_logger().info("🔊 Beep 시작")
    
    def stop_beep(self):
        if self.beep_timer is not None:
            self.beep_timer.cancel()
            self.beep_timer = None
            self.get_logger().info("🔇 Beep 중지")
    
    def beep_callback(self):
        msg = self.create_beep_message()
        if self.robot_selected == 'robot5':
            self.beep_pub_robot5.publish(msg)
        else:
            self.beep_pub_robot3.publish(msg)
    
    # ==========================================
    # 미션 시작 트리거
    # ==========================================
    def try_start_mission(self):
        if not self.mission_started and self.robot_selected and self.target_coordinate:
            self.mission_started = True
            self.get_logger().info("✅ 미션 시작!")
            self.start_undocking()
    
    # ==========================================
    # 단계별 동작 함수들
    # ==========================================
    def start_undocking(self):
        self.publish_progress(0)
        _, dock_nav, namespace = self.get_current_navigator()
        
        if dock_nav.getDockedStatus():
            self.get_logger().info(f"🔓 [{namespace}] 언도킹 시작")
            dock_nav.undock()
            self._undock_timer = self.create_timer(1.0, self.check_undock_complete)
        else:
            self.get_logger().info("이미 언도킹 상태")
            self.on_undock_complete()
    
    def check_undock_complete(self):
        _, dock_nav, _ = self.get_current_navigator()
        if not dock_nav.getDockedStatus():
            self.get_logger().info("✅ 언도킹 완료!")
            if hasattr(self, '_undock_timer'):
                self._undock_timer.cancel()
                delattr(self, '_undock_timer')
            self.on_undock_complete()
    
    def on_undock_complete(self):
        if hasattr(self, '_undock_complete_done'):
            return
        self._undock_complete_done = True
        self.get_logger().info("✅ 언도킹 완료! 안전 직진(50cm) 시도")
        self.move_forward_safety(0.5)

    def move_forward_safety(self, distance, retry_count=0):
        """
        [최적화] 안전 거리 이동 계산
        - Blocking Call(timeout=5.0)을 제거하고, 0.05초만 대기
        - 실패 시 타이머를 이용해 비동기 재시도 (메인 스레드 차단 방지)
        """
        MAX_RETRIES = 5
        
        try:
            tf_buffer = self.get_current_tf_buffer()
            # [최적화] timeout을 5.0 -> 0.05로 획기적으로 줄임
            if not tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)):
                raise Exception("Transform not ready immediately")

            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(seconds=0)
            )
            
            curr_x = transform.transform.translation.x
            curr_y = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, curr_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            safety_x = curr_x + distance * math.cos(curr_yaw)
            safety_y = curr_y + distance * math.sin(curr_yaw)
            safety_yaw_deg = math.degrees(curr_yaw)
            
            self.get_logger().info(f"🚀 안전 거리 확보 중: {distance}m 전진")
            self.send_navigation_goal(safety_x, safety_y, safety_yaw_deg, 
                                    callback=self.move_to_target_first)
                                    
        except Exception as e:
            if retry_count < MAX_RETRIES:
                self.get_logger().warn(f"⚠️ TF 조회 지연 (재시도 {retry_count+1}/{MAX_RETRIES})...")
                # Blocking하지 않고 0.5초 뒤에 다시 이 함수를 호출하도록 예약
                self._tf_retry_timer = self.create_timer(
                    0.5, 
                    lambda: self._retry_safety_move(distance, retry_count + 1)
                )
            else:
                self.get_logger().error(f"❌ 안전 직진 계산 실패(TF Timeout): {e}. 바로 목표 이동.")
                self.move_to_target_first()

    def _retry_safety_move(self, distance, retry_count):
        """재시도용 콜백 (일회성 타이머)"""
        if hasattr(self, '_tf_retry_timer'):
            self._tf_retry_timer.cancel()
            delattr(self, '_tf_retry_timer')
        self.move_forward_safety(distance, retry_count)
    
    def move_to_target_first(self):
        if hasattr(self, '_first_move_started'):
            return
        self._first_move_started = True
        self.publish_progress(1)
        
        target_x, target_y = self.target_coordinate
        approach_x, approach_y, approach_yaw = self.calculate_approach_position(
            target_x, target_y, OFFSET_DISTANCE
        )
        
        self.get_logger().info("🎯 [1차] 목표로 이동 시작")
        self.start_beep()
        self.send_navigation_goal(approach_x, approach_y, approach_yaw, 
                                 callback=self.on_first_arrival)
    
    def on_first_arrival(self):
        if hasattr(self, '_first_arrival_done'):
            return
        self._first_arrival_done = True
        self.get_logger().info("✅ 1차 목표 도착!")
        self._first_arrival_timer = self.create_timer(5.0, self.move_to_rescue_position)

    def move_to_rescue_position(self):
        if hasattr(self, '_rescue_move_started'):
            return
        self._rescue_move_started = True
        self.publish_progress(2)
        x, y, yaw = RESCUE_POSITION
        self.get_logger().info("🏥 Rescue position으로 이동")
        self.send_navigation_goal(x, y, yaw, callback=self.on_rescue_position_arrival)
    
    def on_rescue_position_arrival(self):
        if hasattr(self, '_rescue_arrival_done'):
            return
        self._rescue_arrival_done = True
        self.get_logger().info("✅ Rescue position 도착! 구급대원 대기 중...")
        self.publish_progress(3)
        self.waiting_for_rescue = True
        self.rescue_detected = False
    
    def move_to_target_second(self):
        if hasattr(self, '_second_move_started'):
            return
        self._second_move_started = True
        self.publish_progress(4)
        
        target_x, target_y = self.target_coordinate
        approach_x, approach_y, approach_yaw = self.calculate_approach_position(
            target_x, target_y, OFFSET_DISTANCE
        )
        
        self.get_logger().info("🎯 [2차] 목표로 재이동")
        self.send_navigation_goal(approach_x, approach_y, approach_yaw,
                                 callback=self.on_second_arrival)
    
    def on_second_arrival(self):
        if hasattr(self, '_second_arrival_done'):
            return
        self._second_arrival_done = True
        self.stop_beep()
        self.get_logger().info("✅ 2차 목표 도착! 중단 명령 대기 중...")
        self.waiting_for_stop = True
    
    def move_to_dock(self):
        if hasattr(self, '_dock_move_started') and self._dock_move_started:
            return
        self._dock_move_started = True
        self.publish_progress(5)
        
        _, _, namespace = self.get_current_navigator()
        if namespace == 'robot5':
            dock_x, dock_y, dock_yaw = DOCK_APPROACH_POSITION_ROBOT5
        else:
            dock_x, dock_y, dock_yaw = DOCK_APPROACH_POSITION_ROBOT3
        
        self.get_logger().info("🚉 도킹 스테이션으로 복귀 시작")
        self.send_navigation_goal(dock_x, dock_y, dock_yaw, callback=self.start_docking)
    
    def start_docking(self):
        if hasattr(self, '_docking_started'):
            return
        self._docking_started = True
        self.publish_progress(0)
        
        _, dock_nav, _ = self.get_current_navigator()
        self.get_logger().info("🔌 도킹 시작")
        dock_nav.dock()
        self._dock_timer = self.create_timer(1.0, self.check_dock_complete)

    def check_dock_complete(self):
        _, dock_nav, _ = self.get_current_navigator()
        if dock_nav.getDockedStatus():
            self.get_logger().info("✅ 도킹 완료!")
            if hasattr(self, '_dock_timer'):
                self._dock_timer.cancel()
                delattr(self, '_dock_timer')
            self.on_mission_complete()
    
    def on_mission_complete(self):
        if hasattr(self, '_mission_complete_done'):
            return
        self._mission_complete_done = True
        self.get_logger().info("✅ ===== 미션 완료! =====")
        self.publish_progress(0)
        self.create_timer(3.0, self.reset_mission)
    
    # ==========================================
    # 네비게이션 관련
    # ==========================================
    def send_navigation_goal(self, x, y, yaw_degrees, callback=None):
        nav, _, _ = self.get_current_navigator()
        self.current_goal_pose = self.create_pose_stamped(x, y, yaw_degrees)
        self.is_moving = True
        self.is_paused = False
        self._nav_complete_callback = callback
        nav.goToPose(self.current_goal_pose)
        self.get_logger().info(f"🚀 이동 시작: ({x:.2f}, {y:.2f}, {yaw_degrees:.1f}°)")
    
    def check_navigation_status(self):
        if not self.is_moving:
            return
        
        nav, _, _ = self.get_current_navigator()
        if nav.isTaskComplete():
            result = nav.getResult()
            self.is_moving = False
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info("✅ 네비게이션 성공!")
                if self._nav_complete_callback:
                    callback = self._nav_complete_callback
                    self._nav_complete_callback = None
                    callback()
            else:
                self.get_logger().error(f"❌ 네비게이션 실패: {result}")
    
    # ==========================================
    # 유틸리티 함수들
    # ==========================================
    def get_current_navigator(self):
        if self.robot_selected == 'robot5':
            return self.robot5_nav, self.robot5_dock_nav, 'robot5'
        else:
            return self.robot3_nav, self.robot3_dock_nav, 'robot3'
    
    def get_current_tf_buffer(self):
        return self.tf_buffer_robot5 if self.robot_selected == 'robot5' else self.tf_buffer_robot3
    
    def create_pose_stamped(self, x, y, yaw_degrees):
        nav, _, _ = self.get_current_navigator()
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = nav.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        yaw_rad = math.radians(yaw_degrees)
        q = quaternion_from_euler(0, 0, yaw_rad)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        return goal
    
    def calculate_approach_position(self, target_x, target_y, offset_distance):
        """
        [최적화] TF Lookup Timeout 단축 (5.0s -> 0.1s)
        이 함수는 호출 시점에 즉시 값을 반환해야 하므로 재시도 로직 대신
        실패 시 기본값(Target 좌표 그대로)을 사용하여 Blocking을 방지합니다.
        """
        try:
            tf_buffer = self.get_current_tf_buffer()
            # Timeout을 매우 짧게 주어 메인 루프가 멈추지 않게 함
            transform = tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(seconds=0),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            
            dx = target_x - robot_x
            dy = target_y - robot_y
            yaw_to_target = math.atan2(dy, dx)
            
            approach_x = target_x - offset_distance * math.cos(yaw_to_target)
            approach_y = target_y - offset_distance * math.sin(yaw_to_target)
            approach_yaw_deg = math.degrees(yaw_to_target)
            
            return approach_x, approach_y, approach_yaw_deg
        except Exception as e:
            self.get_logger().error(f"❌ 접근 위치 계산 실패(TF 누락): {e}, Offset 없이 이동합니다.")
            return target_x, target_y, 0.0
    
    def publish_progress(self, state_code):
        msg = Int32()
        msg.data = state_code
        self.progress_pub.publish(msg)
    
    def reset_mission(self):
        if hasattr(self, '_reset_started'):
            return
        self._reset_started = True
        self.get_logger().info("🔄 미션 리셋...")
        self.stop_beep()
        self.cancel_all_timers()
        
        self.target_coordinate = None
        self.robot_selected = None
        self.is_moving = False
        self.is_paused = False
        self.current_goal_pose = None
        self.mission_started = False
        self.waiting_for_rescue = False
        self.rescue_detected = False
        self.waiting_for_stop = False
        self._nav_complete_callback = None
        
        for attr in ['_undock_complete_done', '_first_move_started', '_first_arrival_done',
                    '_rescue_move_started', '_rescue_arrival_done', '_second_move_started',
                    '_second_arrival_done', '_dock_move_started', '_docking_started',
                    '_mission_complete_done', '_reset_started']:
                if hasattr(self, attr):
                    delattr(self, attr)


def main():
    rclpy.init()
    node = RescueRobotEventBased()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()