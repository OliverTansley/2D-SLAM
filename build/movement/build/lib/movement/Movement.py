import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist, Pose2D


class move(Node):

    def __init__(self) -> None:

        super().__init__("minimal_subscriber")
        self.publisher_ = self.create_subscription(Pose2D,'/goal_positions')

        super().__init__('minimal_publisher')
        
        self.targetPos = (1,-2)

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.5  # seconds
        
        self.timer = self.create_timer(timer_period, self.go_2_goal)
        rclpy.spin(self)


    def go_2_goal(self):
        
        distance = math.sqrt((EKF.system_state[0]-self.targetPos[0])**2 + (EKF.system_state[1] - self.targetPos[1])**2)
        angle = math.atan((self.targetPos[0]-EKF.system_state[0])/(self.targetPos[1] - EKF.system_state[1]) if (self.targetPos[1] - EKF.system_state[1]) != 0 else float('inf'))
        if angle < 0:
            angle += 2*math.pi
        magnitude = 1 -1/math.exp(distance)
        speed = Twist()
        
        speed.linear.x = magnitude*math.cos(angle)
        speed.linear.y = magnitude*math.sin(angle)
        speed.angular.z = 1 -1/math.exp(angle - rAngle)

        self.publisher_.publish(speed)
        
    
    def update_targetPos(self,nx,ny):
        self.targetPos = (nx,ny)