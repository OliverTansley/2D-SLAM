import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist, PoseArray, Pose2D


class move(Node):

    plan = []

    def __init__(self) -> None:

        super().__init__("minimal_subscriber")
        self.path_subscriber = self.create_subscription(PoseArray,'goal_positions',move.update_path,10)
        self.state_subscriber = self.create_subscription(Pose2D,'robot_position',lambda msg : self.go_2_goal(msg))

        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
    
    def update_path(msg:PoseArray):
        for pose in msg.poses:
            move.plan.append((pose.x,pose.y))


    def go_2_goal(self,msg):
        for pose in move.plan:
            targetPos = (pose.x,pose.y)
            distance = math.sqrt((msg.x-targetPos[0])**2 + (msg.y - targetPos[1])**2)
            angle = math.atan((targetPos[0]-msg.x)/(self.targetPos[1] - msg.y) if (targetPos[1] - msg.y) != 0 else float('inf'))
        
            if angle < 0:
                angle += 2*math.pi
            magnitude = 1 -1/math.exp(distance)

            speed = Twist()
            speed.linear.x = magnitude*math.cos(angle)
            speed.linear.y = magnitude*math.sin(angle)
            speed.angular.z = 1 -1/math.exp(angle - msg.theta)

            self.publisher_.publish(speed)
        
    
    def update_targetPos(self,nx,ny):
        self.targetPos = (nx,ny)

