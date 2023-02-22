import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist, PoseArray, Pose
from time import sleep

class move(Node):

    robot_state = None
    plan = []

    def __init__(self) -> None:

        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        super().__init__("minimal_subscriber")
        self.path_subscriber = self.create_subscription(PoseArray,'/goal_positions',lambda msg : self.common_callback(msg),10)
        self.state_subscriber = self.create_subscription(Pose,'/robot_position',lambda msg : self.common_callback(msg),10)
        rclpy.spin(self)
        

    def common_callback(self,msg) -> None:
        if isinstance(msg,PoseArray):
            self.update_path(msg)
        
        if isinstance(msg,Pose):
            self.update_state(msg)


    def update_state(self,msg):
        self.robot_state = msg
        if self.plan != []:
            self.go_2_goal((3,5))


    def update_path(self,msg):
        self.get_logger().info("Path recieved")
        self.plan = []
        for pose in msg.poses:
            self.plan.append((pose.orientation.x,pose.orientation.y))
        self.plan.reverse()
        


    def go_2_goal(self,targetPos):
        if self.robot_state != None:
           
            distance = math.sqrt((self.robot_state.orientation.x-targetPos[0])**2 + (self.robot_state.orientation.y - targetPos[1])**2)
            if targetPos[1] - self.robot_state.orientation.y == 0:
                angle = math.atan((targetPos[0]-self.robot_state.orientation.x)*float('inf'))
            else:
                angle = math.atan((targetPos[0]-self.robot_state.orientation.x)/(targetPos[1] - self.robot_state.orientation.y))
        
            # if targetPos[0] < self.robot_state.orientation.x:
            #     if targetPos[1] < self.robot_state.orientation.y:
            #         angle += math.pi
            #     else:
            #         angle += 3*math.pi/2
            # else:
            #     if targetPos[1] < self.robot_state.orientation.y:
            #         angle += math.pi/2
            #     else:
            #         angle += 0

            

            print(angle - self.robot_state.orientation.z)

            magnitude = 0.5 -0.5/math.exp(distance)
            
            speed = Twist()

            if abs(angle - self.robot_state.orientation.z) < 0.6:
                speed.linear.x = magnitude
            
            if angle == float('inf'):
                speed.angular.z = 0.0
            else:
                speed.angular.z = 0.2 -0.2/math.exp(self.robot_state.orientation.z - angle)
            
            
            if distance < 0.3:
                self.plan.remove(targetPos)

            if self.plan == []:
                speed = Twist()
                speed.linear.x = 0.0
                speed.linear.y = 0.0
                speed.angular.z = 0.0
            self.publisher_.publish(speed)

        
    
    def update_targetPos(self,nx,ny):
        self.targetPos = (nx,ny)


if __name__ == '__main__':
    rclpy.init()
    mv = move()