import sys
import os
import sqlite3
import datetime
import time
import threading
import cv2
import numpy as np
import json
import math
import re
import webbrowser
from threading import Timer
from glob import glob

# --- [Flask & ROS2 라이브러리] ---
from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify, flash
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# =========================================================
# [설정] 사용자 환경 및 상수
# =========================================================
CAM_LEFT_ID = 0   
CAM_RIGHT_ID = 2
USER_HOME = os.path.expanduser('~')

# 모델 및 DB 경로
YOLO_MODEL_PATH = os.path.join(USER_HOME, '/home/rokey/turtlebot4_ws/src/subway_control/result04.pt')
DB_NAME = os.path.join(USER_HOME, '/home/rokey/turtlebot4_ws/src/subway_control/subway_control/subway_log.db')

CONF_THRESHOLD = 0.5
DIST_THRESHOLD = 400.0  # 도착 판정 거리

CLASS_ID_PATIENT = 2
CLASS_ID_RESPONDER = 3
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# =========================================================

package_name = 'subway_control'
try:
    template_dir = os.path.join(get_package_share_directory(package_name), 'templates')
    app = Flask(__name__, template_folder=template_dir)
except Exception:
    app = Flask(__name__, template_folder='templates')

app.secret_key = 'subway_secret_key'

robots_data = {
    "robotA": {"bat": 0, "x": 0.0, "y": 0.0, "status": "연결 대기"}
}

# [좌표 버퍼]
target_buffer = {
    "yolo": {
        "valid": False, "u": 0, "v": 0, "x": 0.0, "y": 0.0, "last_seen": 0
    },
    "manual": {
        "valid": False, "u": 0, "v": 0, "x": 0.0, "y": 0.0
    }
}

camera_status = {1: False, 2: False} 
global_frame_left = None
global_frame_right = None
frame_lock = threading.Lock()

# =========================================================
# [DB 관리자]
# =========================================================
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db_dir = os.path.dirname(DB_NAME)
    if not os.path.exists(db_dir): 
        try:
            os.makedirs(db_dir)
        except Exception:
            pass
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS emergency_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            content TEXT, 
            log_history TEXT, 
            timestamp TEXT,
            patient_name TEXT, 
            patient_gender TEXT, 
            patient_age TEXT,
            patient_location TEXT, 
            patient_status TEXT, 
            remarks TEXT
        )
    """)
    c.execute("CREATE TABLE IF NOT EXISTS robot_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, timestamp TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    
    cols = ['log_history', 'patient_name', 'patient_gender', 'patient_age', 'patient_location', 'patient_status', 'remarks']
    for col in cols:
        try:
            c.execute(f"ALTER TABLE emergency_history ADD COLUMN {col} TEXT")
        except Exception:
            pass

    try:
        c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", ('rokey', 'rokey1234'))
        conn.commit()
    except Exception:
        pass
    conn.close()
    print(f">>> [DB] Database Initialized at: {DB_NAME}")

def parse_time_safe(time_str):
    try:
        return datetime.datetime.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.datetime.now()

def save_accumulated_log(source, message_raw):
    conn = get_db_connection()
    c = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
    
    if not message_raw.startswith("["):
        message = f"[{source}] {message_raw}"
    else:
        message = message_raw

    c.execute("SELECT * FROM emergency_history ORDER BY id DESC LIMIT 1")
    last_row = c.fetchone()
    
    new_log_entry = f"[{now_str}] {message}"
    should_insert_new = True
    
    if last_row:
        try:
            last_dt = parse_time_safe(last_row['timestamp'])
            if (datetime.datetime.now() - last_dt).total_seconds() < 1800:
                should_insert_new = False
                existing = last_row['log_history'] if last_row['log_history'] else ""
                last_line = existing.strip().split('\n')[-1] if existing else ""
                
                if message not in last_line:
                    updated = existing + "\n" + new_log_entry
                    c.execute("UPDATE emergency_history SET content=?, log_history=?, timestamp=? WHERE id=?", 
                              (message, updated, now_str, last_row['id']))
                    print(f"💾 [DB Updated] {message}")
        except Exception:
            pass
            
    if should_insert_new:
        c.execute("INSERT INTO emergency_history (content, log_history, timestamp) VALUES (?, ?, ?)", 
                  (message, new_log_entry, now_str))
        print(f"💾 [DB New Insert] {message}")
    
    conn.commit()
    conn.close()

def save_simple_log(content):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
        c.execute("INSERT INTO robot_logs (content, timestamp) VALUES (?, ?)", (content, now))
        conn.commit()
        conn.close()
    except Exception:
        pass

# =========================================================
# [Vision & Homography]
# =========================================================
class HomographyConverter:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.H = None
        self.init_matrix()

    def init_matrix(self):
        if self.cam_id == CAM_LEFT_ID: 
            pixel = np.array([[329, 241], [949, 214], [1242, 594], [137, 702]], dtype=np.float32)
            map_pt = np.array([[-0.408, 2.433], [-0.003, -0.002], [-2.200, -0.356], [-2.650, 2.096]], dtype=np.float32)
        elif self.cam_id == CAM_RIGHT_ID: 
            pixel = np.array([[455, 95], [819, 91], [1225, 658], [45, 647]], dtype=np.float32)
            map_pt = np.array([[2.850, -0.643], [3.275, -3.795], [-1.632, -4.550], [-1.997, -1.627]], dtype=np.float32)
        else:
            return
        self.H, _ = cv2.findHomography(pixel, map_pt)

    def pixel_to_map(self, u, v):
        if self.H is None:
            return 0.0, 0.0
        p = np.array([[[u, v]]], dtype=np.float32)
        m = cv2.perspectiveTransform(p, self.H)
        return float(m[0][0][0]), float(m[0][0][1])

class VisionSystem(threading.Thread):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.running = True
        self.daemon = True
        self.br = CvBridge()
        
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
        except Exception: 
            print(f"❌ YOLO 모델 로드 실패: {YOLO_MODEL_PATH}")
            self.running = False
            
        self.conv_l = HomographyConverter(CAM_LEFT_ID)
        self.conv_r = HomographyConverter(CAM_RIGHT_ID)
        self.last_auto_action_time = 0
        
        # [플래그] 도착 신호 중복 전송 방지
        self.is_arrival_sent = False

    def init_camera(self, idx):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        return cap if cap.isOpened() else None

    def run(self):
        global global_frame_left, global_frame_right, camera_status
        c_l = self.init_camera(CAM_LEFT_ID)
        c_r = self.init_camera(CAM_RIGHT_ID)
        
        while self.running:
            r_l, f_l = c_l.read() if c_l else (False, None)
            r_r, f_r = c_r.read() if c_r else (False, None)
            camera_status[1], camera_status[2] = r_l, r_r

            if not r_l and not r_r:
                time.sleep(1)
                continue
            
            frames = []
            if r_l: 
                frames.append(f_l)
                #if self.ros_node:
                #    self.ros_node.pub_video.publish(self.br.cv2_to_imgmsg(f_l, encoding='bgr8'))
            if r_r:
                frames.append(f_r)
            
            if frames and hasattr(self, 'model'):
                try:
                    res = self.model(frames, conf=CONF_THRESHOLD, verbose=False)
                    i = 0
                    if r_l: 
                        self.proc(res[i], f_l, self.conv_l)
                        with frame_lock:
                            global_frame_left = f_l.copy()
                        i += 1
                    if r_r: 
                        self.proc(res[i], f_r, self.conv_r)
                        with frame_lock:
                            global_frame_right = f_r.copy()
                except Exception:
                    pass
            time.sleep(0.01)

        if c_l: c_l.release()
        if c_r: c_r.release()

    def proc(self, res, frame, conv):
        pc, rc = None, None
        for box in res.boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            col = (0, 255, 0) if cid == CLASS_ID_RESPONDER else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, self.model.names[cid], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            
            if cid == CLASS_ID_PATIENT:
                pc = (cx, cy)
                mx, my = conv.pixel_to_map(cx, cy)
                
                target_buffer["yolo"] = {
                    "valid": True, "u": cx, "v": cy, 
                    "x": mx, "y": my, "last_seen": time.time()
                }

                current_time = time.time()
                if current_time - self.last_auto_action_time > 3.0:
                    self.last_auto_action_time = current_time
                    save_simple_log(f"[YOLO] 환자 감지됨! 대기 중... ({mx:.1f}, {my:.1f})")

            elif cid == CLASS_ID_RESPONDER: 
                rc = (cx, cy)
        
        # [도착 로직] 거리 400 이하일 때 5회 전송 & 플래그 처리
        if pc and rc:
            d = math.sqrt((pc[0] - rc[0])**2 + (pc[1] - rc[1])**2)
            
            if d <= DIST_THRESHOLD: # 거리 400 기준
                if not self.is_arrival_sent and self.ros_node:
                    print(f"도착 완료(거리: {d:.1f}) -> 신호 전송 시작")
                    save_simple_log(f"구급대원-환자 접촉 확인 (거리: {d:.1f}) -> 도착 신호 전송")
                    for _ in range(5):
                        self.ros_node.pub_arrival.publish(Bool(data=True))
                        time.sleep(0.05)
                    self.is_arrival_sent = True 
            else:
                self.is_arrival_sent = False

# =========================================================
# [ROS2 Node]
# =========================================================
class ControlTowerNode(Node):
    def __init__(self):
        super().__init__('subway_control_tower')
        self.pub_target = self.create_publisher(PoseStamped, '/target', 10)
        self.pub_task_end = self.create_publisher(Bool, '/stop', 10)
        self.pub_arrival = self.create_publisher(Bool, '/emt_arrival_status', 10)
       # self.pub_video = self.create_publisher(Image, 'video_frames', 10)
        
        self.last_msg_a = None
        self.last_msg_b = None
        
        self.create_subscription(String, '/robotA/task_progress', self.cb_rob_a, 10)
        self.create_subscription(String, '/robotB/task_progress', self.cb_rob_b, 10)
        self.create_subscription(String, '/system/alert', self.cb_sys, 10)

    def cb_rob_a(self, msg):
        if self.last_msg_a == msg.data:
            return
        self.last_msg_a = msg.data
        save_accumulated_log("Robot A", msg.data)

    def cb_rob_b(self, msg):
        if self.last_msg_b == msg.data:
            return
        self.last_msg_b = msg.data
        save_accumulated_log("Robot B", msg.data)

    def cb_sys(self, msg):
        try:
            d = json.loads(msg.data)
            if isinstance(d, dict):
                robots_data["robotA"].update(d)
            else:
                robots_data["robotA"]["status"] = str(d)
        except Exception:
            pass

    def pub_goal(self, x, y):
        m = PoseStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.orientation.w = 1.0
        self.pub_target.publish(m)

    def send_task_end_signal(self):
        self.pub_task_end.publish(Bool(data=True))
        save_simple_log("명령 전송: 작업 종료")

# =========================================================
# [Flask API]
# =========================================================
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if c.fetchone():
            session['user'] = u
            conn.close()
            return redirect(url_for('dashboard'))
        conn.close()
        flash("❌ 로그인 실패")
    return render_template('login_center.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users VALUES (?, ?)", (u, p))
            conn.commit()
            conn.close()
            flash("✅ 등록 완료")
            return redirect(url_for('login_page'))
        except Exception:
            conn.close()
            flash("❌ ID 중복")
            return redirect(url_for('signup_page'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('sysmon.html', username=session.get('user', 'Guest'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

@app.route('/history')
def history_page():
    return render_template('history.html', username=session.get('user', 'Guest'))

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', username=session.get('user', 'Guest'))

@app.route('/api/history/list')
def get_history_list():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, timestamp, content, log_history FROM emergency_history ORDER BY id DESC")
    rows = [dict(row) for row in c.fetchall()]
    
    total = len(rows)
    safe_days = 0
    if rows:
        try:
            safe_days = (datetime.datetime.now() - parse_time_safe(rows[0]['timestamp'])).days
        except Exception:
            pass
    else:
        safe_days = 365
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(*) as cnt FROM robot_logs WHERE timestamp LIKE ?", (f"{today}%",))
    today_cmds_row = c.fetchone()
    today_cmds = today_cmds_row['cnt'] if today_cmds_row else 0
    
    conn.close()
    return jsonify({
        "history": rows, 
        "stats": {
            "total_cases": total, 
            "safe_days": safe_days, 
            "total_cmds": f"{today_cmds:,}"
        }
    })

@app.route('/api/status')
def get_status_api():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM emergency_history ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    
    emer_logs = []
    active = False
    if row:
        try:
            if (datetime.datetime.now() - parse_time_safe(row['timestamp'])).total_seconds() < 1800:
                active = True
            if row['log_history']:
                emer_logs = [[row['id'], l, ""] for l in row['log_history'].split('\n')]
        except Exception:
            pass
            
    c.execute("SELECT * FROM robot_logs ORDER BY id DESC LIMIT 50")
    sys_logs = [[r['id'], r['content'], r['timestamp']] for r in c.fetchall()]
    conn.close()
    
    yolo_act = False
    if target_buffer["yolo"]["valid"]:
        if time.time() - target_buffer["yolo"]["last_seen"] < 2.0:
            yolo_act = True
        else:
            target_buffer["yolo"]["valid"] = False

    return jsonify({
        "robots": {"robotA": robots_data["robotA"]}, 
        "logs": {"emergency": emer_logs[::-1], "system": sys_logs}, 
        "is_active": active,
        "targets": {
            "yolo": target_buffer["yolo"], 
            "manual": target_buffer["manual"], 
            "yolo_active": yolo_act
        }
    })

@app.route('/api/analytics/data')
def get_analytics_data():
    tid = request.args.get('id')
    conn = get_db_connection()
    c = conn.cursor()
    q = "SELECT * FROM emergency_history WHERE id=?" if tid else "SELECT * FROM emergency_history ORDER BY id DESC LIMIT 1"
    c.execute(q, (tid,) if tid else ())
    row = c.fetchone()
    
    logs = []
    patient = {}
    if row:
        patient = {
            "name": row['patient_name'], "gender": row['patient_gender'], 
            "age": row['patient_age'], "location": row['patient_location'], 
            "status": row['patient_status'], "remarks": row['remarks']
        }
        if row['log_history']:
            for l in row['log_history'].split('\n'):
                try:
                    parts = l.split('] ', 1)
                    logs.append([0, parts[1], parts[0][1:]])
                except Exception:
                    logs.append([0, l, ""])
    conn.close()
    return jsonify({"logs": logs, "patient": patient})

@app.route('/api/analytics/update', methods=['POST'])
def update_analytics():
    d = request.json
    rid = d.get('id')
    conn = get_db_connection()
    c = conn.cursor()
    
    if not rid:
        c.execute("SELECT id FROM emergency_history ORDER BY id DESC LIMIT 1")
        last = c.fetchone()
        if last:
            rid = last['id']
        else:
            conn.close()
            return jsonify({"status":"error"}), 404
            
    c.execute("""
        UPDATE emergency_history 
        SET patient_name=?, patient_gender=?, patient_age=?, 
            patient_location=?, patient_status=?, remarks=? 
        WHERE id=?
    """, (d.get('name'), d.get('gender'), d.get('age'), d.get('location'), 
          d.get('status'), d.get('remarks'), rid))
    
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

# =========================================================
# [클릭 이벤트] 수동 모드(Manual)일 때만 즉시 토픽 전송! (5회)
# =========================================================
@app.route('/api/click', methods=['POST'])
def click_event():
    d = request.json
    print(f"DEBUG: 웹 데이터 수신: {d}")

    # [수정] 모드 정보가 없으면 'manual'로 기본 설정
    current_mode = d.get('mode', 'manual')
    
    if current_mode != 'manual':
        print(f"DEBUG: Manual 모드가 아니라서 무시함 (Mode: {current_mode})")
        return jsonify({"status": "ignored", "msg": "Only available in Manual Mode"})

    try:
        cid = int(d.get('id', 0))
        u = int(d.get('x', 0))
        v = int(d.get('y', 0))
    except:
        return jsonify({"status": "error"})

    if vision_system:
        # 1. 호모그래피 변환
        conv = vision_system.conv_l if cid == 1 else vision_system.conv_r
        mx, my = conv.pixel_to_map(u, v)
        
        # 2. 좌표 기록
        target_buffer["manual"] = {
            "valid": True, "u": u, "v": v, "x": mx, "y": my
        }
        save_simple_log(f"[Manual] 클릭 좌표: ({mx:.1f}, {my:.1f}) -> 이동 시작")
        
        # 3. [핵심] 수동 클릭 즉시 5회 반복 전송!
        if ros_node:
            print(f"Manual Direct Dispatch: {mx}, {my} (5회 전송)")
            for _ in range(5):
                ros_node.pub_goal(mx, my)
                time.sleep(0.05)
        
    return jsonify({"status": "success"})

# =========================================================
# [출동 버튼] 오토 모드용 (5회 전송)
# =========================================================
@app.route('/api/dispatch', methods=['POST'])
def dispatch_robot():
    d = request.json
    mode = d.get('mode')
    
    tx, ty = 0.0, 0.0
    valid = False
    src = ""
    
    if mode == 'yolo' and target_buffer['yolo']['valid']:
        tx, ty = target_buffer['yolo']['x'], target_buffer['yolo']['y']
        valid = True
        src = "AUTO(YOLO)"
    elif mode == 'manual' and target_buffer['manual']['valid']:
        tx, ty = target_buffer['manual']['x'], target_buffer['manual']['y']
        valid = True
        src = "MANUAL(BTN)"
        
    if valid and ros_node:
        print(f"버튼 출동 명령: {src} -> {tx}, {ty}")
        for _ in range(5): 
            ros_node.pub_goal(tx, ty)
            time.sleep(0.05)
        save_simple_log(f"로봇 출동 ({src}) -> ({tx:.2f}, {ty:.2f})")
        return jsonify({"status": "success"})
    
    return jsonify({"status": "error"}), 400

@app.route('/api/task_end', methods=['POST'])
def task_end():
    if ros_node:
        ros_node.send_task_end_signal()
    return jsonify({"status": "success"})

def gen(t):
    global global_frame_left, global_frame_right
    while True:
        f = global_frame_left if t=='L' else global_frame_right
        with frame_lock:
            if f is None:
                _, buf = cv2.imencode('.jpg', np.zeros((720, 1280, 3), np.uint8))
            else:
                _, buf = cv2.imencode('.jpg', f)
        
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)

@app.route('/video/1')
def v1():
    return Response(gen('L'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/2')
def v2():
    return Response(gen('R'), mimetype='multipart/x-mixed-replace; boundary=frame')

ros_node = None
vision_system = None

def main(args=None):
    global ros_node, vision_system
    init_db()
    rclpy.init(args=args)
    ros_node = ControlTowerNode()
    vision_system = VisionSystem(ros_node)
    
    threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True).start()
    vision_system.start()
    
    Timer(1.5, lambda: webbrowser.open_new('http://localhost:5000') if not os.environ.get("WERKZEUG_RUN_MAIN") else None).start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception:
        pass
    finally:
        vision_system.running = False
        vision_system.join()
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()