import sqlite3
import random
from datetime import datetime, timedelta

# === [설정] 리얼 데이터를 위한 샘플 목록 ===
LAST_NAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전"]
FIRST_NAMES = ["민수", "서준", "도윤", "예준", "시우", "하준", "지호", "주원", "지후", "준우", "서연", "서윤", "지우", "서현", "하은", "민서", "지유", "윤서", "채원", "수아", "철수", "영희", "길동", "진수", "미영", "석진"]
LOCATIONS = ["B1F 대합실", "1-1 승강장", "4-3 승강장", "B2F 환승 통로", "화장실 앞", "에스컬레이터 하단", "3번 게이트 앞", "편의점 앞", "엘리베이터 입구"]
STATUSES = ["심정지 의심", "의식 불명", "호흡 곤란", "흉통 호소", "낙상 사고", "어지러움 호소", "두부 외상"]
REMARKS = ["보호자 동행함", "특이사항 없음", "자동제세동기(AED) 패드 부착", "심폐소생술(CPR) 실시함", "구급대원에게 환자 병력 전달함", "초기 의식 있었으나 소실됨"]

def generate_random_name():
    return random.choice(LAST_NAMES) + random.choice(FIRST_NAMES)

def reset_and_fill_data(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print(f"[{db_path}] 기존 데이터 초기화 및 A/B 협업 데이터 생성 중...")

    # 1. 기존 테이블 삭제 (초기화)
    c.execute("DROP TABLE IF EXISTS emergency_history")
    c.execute("DROP TABLE IF EXISTS robot_logs")
    c.execute("DROP TABLE IF EXISTS users")

    # 2. 테이블 새로 생성
    c.execute("""
        CREATE TABLE emergency_history (
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
    
    c.execute("""
        CREATE TABLE robot_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            content TEXT, 
            timestamp TEXT
        )
    """)

    c.execute("CREATE TABLE users (username TEXT PRIMARY KEY, password TEXT)")
    c.execute("INSERT INTO users VALUES ('rokey', 'rokey1234')")

    # 3. 6년치 데이터 생성
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    for year in years:
        count = random.randint(4, 8)
        for _ in range(count):
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            hour = random.randint(5, 23)
            minute = random.randint(0, 59)
            
            base_time = datetime(year, month, day, hour, minute)
            
            # [타임라인] A와 B가 섞인 시나리오 생성
            t1 = base_time
            t2 = t1 + timedelta(seconds=random.randint(20, 40))
            t3 = t2 + timedelta(seconds=random.randint(30, 60))
            t4 = t3 + timedelta(seconds=random.randint(40, 80))
            t5 = t4 + timedelta(seconds=random.randint(50, 100))
            t6 = t5 + timedelta(seconds=random.randint(60, 120))
            
            ts1 = t1.strftime("%Y-%m-%d %H:%M:%S")
            ts2 = t2.strftime("%Y-%m-%d %H:%M:%S")
            ts3 = t3.strftime("%Y-%m-%d %H:%M:%S")
            ts4 = t4.strftime("%Y-%m-%d %H:%M:%S")
            ts5 = t5.strftime("%Y-%m-%d %H:%M:%S")
            ts6 = t6.strftime("%Y-%m-%d %H:%M:%S")
            
            loc = random.choice(LOCATIONS)
            
            # [중요] analytics.html 키워드(AED, 통제, 구급대, 복귀)에 맞춘 로그
            log_hist = (
                f"[{ts1}] [System] 응급 상황 알림 수신 (위치: {loc}) - 상황 접수\n"
                f"[{ts2}] [Robot A] AED 이송 시작\n"
                f"[{ts3}] [Robot B] 환자위치로 이동하여 상태 파악\n"
                f"[{ts4}] [Robot A] 현장 도착 및 승객 통제 실시\n"
                f"[{ts5}] [Robot B] 119 구급대원과 함께 환자위치로 이동\n"
                f"[{ts6}] [Robot A] 상황 종료. 도킹 스테이션 복귀"
            )
            
            p_name = generate_random_name()
            p_gender = random.choice(["M", "F"])
            p_age = f"{random.randint(20, 80)}세"
            p_stat = random.choice(STATUSES)
            p_remark = random.choice(REMARKS)
            
            c.execute("""
                INSERT INTO emergency_history (
                    content, timestamp, log_history, 
                    patient_name, patient_gender, patient_age, 
                    patient_location, patient_status, remarks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "상황 종료 (구급대 인계 완료)", 
                ts6 + ".123", 
                log_hist,
                p_name, p_gender, p_age, loc, p_stat, p_remark
            ))

        # 4. 평시 로그 (Robot A, B 섞어서)
        for m in range(1, 13):
            for _ in range(10):
                log_ts = datetime(year, m, random.randint(1, 28), random.randint(6, 23)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                robot = "Cam1" if random.random() > 0.5 else "Cam2"
                c.execute("INSERT INTO robot_logs (content, timestamp) VALUES (?, ?)",
                          (f"클릭 명령: {robot}->Map({random.uniform(-5,5):.1f},{random.uniform(-5,5):.1f})", log_ts))

    # 5. 금일 통계용 데이터
    today = datetime.now()
    for _ in range(random.randint(10, 20)):
        log_ts = today.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        c.execute("INSERT INTO robot_logs (content, timestamp) VALUES (?, ?)",
                  (f"클릭 명령: Cam{random.randint(1,2)}->Map({random.uniform(-5,5):.1f},{random.uniform(-5,5):.1f})", log_ts))

    conn.commit()
    conn.close()
    print("✅ 모든 데이터 생성 완료! (6년치 + 금일 통계 + A/B 시나리오)")

if __name__ == '__main__':
    reset_and_fill_data('subway_log.db')
