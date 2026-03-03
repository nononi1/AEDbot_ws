import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time

class MultiSpamTester(Node):
    def __init__(self):
        super().__init__('multi_spam_tester')
        self.pub_a = self.create_publisher(String, '/robotA/task_progress', 10)
        self.pub_b = self.create_publisher(String, '/robotB/task_progress', 10)

    def robot_a_logic(self):
        """로봇 A: 'A 상황 발생'을 30번 연속 전송"""
        print("🔵 Robot A: 폭주 시작!")
        for i in range(30):
            msg = String(data="A 상황 발생")
            self.pub_a.publish(msg)
            time.sleep(0.01)
        print("🔵 Robot A: 전송 완료")

    def robot_b_logic(self):
        """로봇 B: 'B 상황 발생'을 30번 연속 전송"""
        print("🟠 Robot B: 폭주 시작!")
        for i in range(30):
            msg = String(data="B 상황 발생")
            self.pub_b.publish(msg)
            time.sleep(0.01)
        print("🟠 Robot B: 전송 완료")

    def run_test(self):
        print("🚀 [Multi-Robot Spam Test] A, B 동시 전송 테스트 시작\n")
        
        # 서버 연결 대기
        while self.pub_a.get_subscription_count() == 0 or self.pub_b.get_subscription_count() == 0:
            print("⏳ 서버 연결 기다리는 중...")
            time.sleep(1)

        print("✅ 서버 연결 확인! 2초 뒤 두 로봇이 동시에 메시지를 보냅니다.")
        time.sleep(2)

        # 쓰레드를 사용하여 A와 B가 동시에 함수를 실행하도록 함
        thread_a = threading.Thread(target=self.robot_a_logic)
        thread_b = threading.Thread(target=self.robot_b_logic)

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()

        print("\n🏁 모든 전송이 끝났습니다. 서버 터미널의 DB 로그를 확인하세요.")

def main(args=None):
    rclpy.init(args=args)
    node = MultiSpamTester()
    node.run_test()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
