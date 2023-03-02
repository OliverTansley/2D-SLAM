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
        # if self.plan != []:
            # self.go_2_goal(self.plan[0])
        self.circle()

    def update_path(self,msg):
        self.get_logger().info("Path recieved")
        self.plan = []
        for pose in msg.poses:
            self.plan.append((pose.orientation.x,pose.orientation.y))
        self.plan.reverse()

    def circle(self):
        speed = Twist()
        speed.linear.x = 0.2
        speed.linear.y = 0.0
        speed.angular.z = 0.1
        self.publisher_.publish(speed)

    def go_2_goal(self,targetPos):
        if self.robot_state != None:
           
            distance = math.sqrt((self.robot_state.orientation.x-targetPos[0])**2 + (self.robot_state.orientation.y - targetPos[1])**2)
            if targetPos[1] - self.robot_state.orientation.y == 0:
                angle = math.atan((targetPos[0]-self.robot_state.orientation.x)*float('inf'))
            else:
                if targetPos[0] < self.robot_state.orientation.x:
                    if targetPos[1] < self.robot_state.orientation.y:
                        angle = math.atan((targetPos[1] - self.robot_state.orientation.y)/(targetPos[0]-self.robot_state.orientation.x)) 
                    else:
                        angle = math.atan((targetPos[1] - self.robot_state.orientation.y)/(targetPos[0]-self.robot_state.orientation.x)) 
                else:
                    if targetPos[1] < self.robot_state.orientation.y:
                        angle = math.atan((targetPos[1] - self.robot_state.orientation.y)/(targetPos[0]-self.robot_state.orientation.x)) + math.pi/2
                    else:
                        angle = math.atan((targetPos[1] - self.robot_state.orientation.y)/(targetPos[0]-self.robot_state.orientation.x)) +math.pi
  
            if angle > 2*math.pi:
                angle -= 2*math.pi
            if angle < 0:
                angle += 2*math.pi
            
            ang_diff = angle - self.robot_state.orientation.z
            
            magnitude = 0.5 -0.5*math.exp(-distance)
            
            speed = Twist()

            if abs(ang_diff) < 0.25:
                speed.linear.x = magnitude
                
            
            speed.angular.z = (ang_diff/abs(ang_diff))*(0.2 -0.2*math.exp(-2*abs(ang_diff)))
            if distance < 0.1:
                self.plan.remove(targetPos)

            if self.plan == []:
                speed.linear.x = 0.0
                speed.linear.y = 0.0
                speed.angular.z = 0.0
            self.publisher_.publish(speed)
        print(targetPos)
        print(self.plan)

        
    
    def update_targetPos(self,nx,ny):
        self.targetPos = (nx,ny)


if __name__ == '__main__':
    rclpy.init()
    mv = move()