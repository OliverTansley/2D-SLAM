from rclpy.node import Node
from std_msgs.msg import Pose2D,String
import rclpy

class Controller(Node):

    def __init__(self) -> None:
        self.__init__('minimal_publisher')
        publisher = self.create_publisher(Pose2D,'start_planning',10)

        self.__init__('minimal_subscriber')
        subscriber = self.create_subscription(String,'finished',lambda msg : self.get_next_instr(msg,publisher),10)

        rclpy.spin(self)

    def get_next_instr(self,msg,publisher):
        if msg.data == 'finished':
            x = input("enter x position")
            y = input("enter y position")
            msg = Pose2D()
            msg.x = x
            msg.y = y
            publisher.publish(msg)

if __name__ == '__main__':
    rclpy.init()
    cr = Controller()