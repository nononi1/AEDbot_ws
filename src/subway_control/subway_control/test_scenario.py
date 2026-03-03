import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class ScenarioTester(Node):
    def __init__(self):
        super().__init__('scenario_tester')
        
        # [핵심] 토픽 이름으로 로봇 구분 (/robotA..., /robotB...)
        self.pub_a = self.create_publisher(String, '/robotA/task_progress', 10)
        self.pub_b = self.create_publisher(String, '/robotB/task_progress', 10)
        
        # [시나리오] (로봇, 보낼 메시지)
        # 메시지 내용에는 로봇 이름 없이 깔끔하게 상태만 넣었습니다.
        self.scenario_steps = [
            # 1. 초기 상태
            ('A', "로봇 대기"),
            ('B', "로봇 대기"),
            
            # 2. 상황 발생 및 출동
            ('A', "AED 이송"),           # A: AED 가지고 출발
            ('B', "환자위치로 이동"),      # B: 환자 위치 파악하러 이동
            
            # 3. 현장 조치 및 통제
            ('A', "승객 통제"),           # A: 도착 후 안내방송 및 통제
            
            # 4. 구급대 인계 과정 (B의 역할)
            ('B', "구급대원 위치로 이동"), # B: 구급대 마중
            ('B', "구급대원 대기"),        # B: 입구에서 대기
            ('B', "구급대원과 함께 환자위치로 이동"), # B: 구급대 인솔
            
            # 5. 상황 종료 및 복귀
            ('A', "도킹 스테이션 복귀"),   
            ('B', "도킹 스테이션 복귀")    
        ]

    def run_scenario(self):
        print("🚀 [Multi-Robot] 시나리오 테스트 시작! (서버 연결 확인 중...)")
        
        # 서버가 켜져 있는지 확인 (둘 다 연결될 때까지 대기)
        while self.pub_a.get_subscription_count() == 0 or self.pub_b.get_subscription_count() == 0:
            print("⏳ 서버 기다리는 중... (control_tower.py 실행 확인 필요)")
            time.sleep(1)
            
        print("✅ 서버 연결됨! 시나리오 전송 시작\n")

        for robot, msg in self.scenario_steps:
            log_msg = String()
            log_msg.data = msg  # 순수 메시지 내용만 전송
            
            if robot == 'A':
                self.pub_a.publish(log_msg)
                print(f"🔵 [To: /robotA/task_progress] 데이터: '{msg}'")
            else:
                self.pub_b.publish(log_msg)
                print(f"🟠 [To: /robotB/task_progress] 데이터: '{msg}'")
            
            # 웹 화면에서 단계 변화를 확인하기 위해 3초 간격 둠
            time.sleep(3) 

        print("\n🏁 시나리오 전송 완료! 웹 화면 로그를 확인하세요.")

def main(args=None):
    rclpy.init(args=args)
    tester = ScenarioTester()
    tester.run_scenario()
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
