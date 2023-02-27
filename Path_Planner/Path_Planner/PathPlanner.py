from DataStructures import Tree, TNode
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from seed_segment import SeedSegment
from geometry_msgs.msg import Pose, PoseArray, Twist
from sensor_msgs.msg import LaserScan
import logging
import random
import math
import matplotlib.pyplot as plt
import rclpy
from time import sleep

def point_2_point_distance(point1: tuple[int,int],point2:tuple[int,int]) -> float:
    '''
    Get euclidean distance between to points represented as tuples (x1,y1) (x2,y2)
    '''
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)


def line_get_intersect(line1,line2):
    '''
    Returns x and y value for the intersection of two lines
    '''
    i_x = (line1[1] - line2[1])/(line2[0] - line1[0])
    i_y = (line1[0]*line2[1] - line1[1]*line2[0])/(line1[0] - line2[0])
    return (i_x,i_y)

class PathPlanner(Node):

    N:int = 40
    found:bool = False
    k:int = 200

    start = (0,0,0)
    end = (0,3)

    lines = []

    def __init__(self) -> None:

        self.poses = []

        super().__init__('minimal_publisher')
        self.path_publisher = self.create_publisher(PoseArray,'/goal_positions',10)
        self.movement_publisher = self.create_publisher(Twist,'/cmd_vel',10)

        super().__init__('minimal_subscriber')
        self.lines_subscription = self.create_subscription(Float32MultiArray,'/line_segments',lambda msg :self.common_callback(msg),10)
        self.position_subscription = self.create_subscription(Pose,'/robot_position',lambda msg :self.common_callback(msg),10)
        self.laserScan_subscription = self.create_subscription(LaserScan,'/scan',lambda msg :self.common_callback(msg),10)

        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-6,6)
        self.ax.set_ylim(-6,6)
        rclpy.spin(self)


    def common_callback(self,msg):
        if isinstance(msg,Float32MultiArray):
            self.update_lines(msg)
            
        if isinstance(msg,Pose):
            self.update_pos(msg)
        if isinstance(msg,LaserScan):
            self.check_replan(msg)
            
            self.lidar_2_points(msg)


    def update_lines(msg):
        PathPlanner.lines = PathPlanner.get_seed_segs(msg)
        

    def update_pos(self,msg):
        self.start = (msg.orientation.x,msg.orientation.y,msg.orientation.z)


    def collision_detected(self,point2,point1,seed_segs) -> bool:

        m = (point2[1]-point1[1])/(point2[0]-point1[0]) if (point2[0]-point1[0]) != 0 else float('inf')
        c = point1[1] - m*point1[0]

        for segment in seed_segs:
            for p in [point2,point1]:
                if point_2_point_distance(p,(segment.x,segment.y)) < 0.45 or self.point_2_line_distance((m,c),(segment.x,segment.y)) < 0.45:  
                    return True  
         
        return False     
        

    def link_lidar(self,lidar_points):
        lines = []
        for i in range(len(lidar_points)-1):
            if point_2_point_distance(lidar_points[i],lidar_points[i+1]) < 200:
                line = PathPlanner.points_2_line(lidar_points[i],lidar_points[i+1])
                lines.append(SeedSegment(line[0],line[1],[lidar_points[i][0]],[lidar_points[i+1][1]]))

        return lines


    def check_replan(self,msg:LaserScan):
        if self.poses == []:
            xs,ys = self.lidar_2_points(msg)
            lidar_points = list(zip(xs,ys))
            
            self.get_path(self.path_publisher,self.link_lidar(lidar_points))
        # else:
        #     xs,ys = self.lidar_2_points(msg)
        #     lidar_points = list(zip(xs,ys))
        #     for i in range(len(self.poses)-2):
        #         if self.collision_detected(((self.poses[i].orientation.x,self.poses[i].orientation.y)),(self.poses[i+1].orientation.x,self.poses[i+1].orientation.y),self.link_lidar(lidar_points)):
        #             self.get_path(self.path_publisher,self.link_lidar(lidar_points))

    def point_2_line_distance(self,line,point):
        '''
        determines a points euclidean distance from the seed segment line
        '''
        
        perpendicular_grad = -1/line[0]
        perpendicular_intercept = point[1] - perpendicular_grad*point[0]

        i_x = (perpendicular_intercept - line[1])/(line[0] - perpendicular_grad)
        i_y = (line[1]*perpendicular_grad - line[0]*perpendicular_intercept)/(perpendicular_grad - line[0])

        return math.sqrt((point[0] - i_x)**2 + (point[1] - i_y)**2)

    @staticmethod
    def get_seed_segs(line_data):
        seed_segs = []
        for i in range(0,len(line_data),6):
            seed_segs.append(SeedSegment.from_zipped_points(line_data[i],line_data[i+1],line_data[i+2],line_data[i+3],line_data[i+4],line_data[i+5]))
        return seed_segs


    def lidar_2_points(self,lidar_msg):
        '''
        Converts raw lidar data to arrays of x and y coordinates relative to scanner
        '''
        lidar_ranges = lidar_msg.ranges
        ang_inc = lidar_msg.angle_increment
        xs = []
        ys = []
       
        for measurement in range(0,len(lidar_ranges)):
            xs.append(self.start[0] - lidar_ranges[measurement] * math.sin(measurement*ang_inc + math.pi/2 + self.start[2]))
            ys.append(self.start[1] + lidar_ranges[measurement] * math.cos(measurement*ang_inc + math.pi/2 + self.start[2]))
        
        self.ax.plot(xs,ys,'b+')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        return xs,ys

    
    def get_path(self,publisher,lidar_segs=[])->None:
        '''
        RRT with euclidean distance heuristic
        '''
        speed = Twist()
        speed.linear.x = 0.0
        speed.linear.y = 0.0
        speed.angular.z = 0.0    
        self.movement_publisher.publish(speed)
        
        msg = PoseArray()
        msg.poses = []
        self.path_publisher.publish(msg)

        self.poses = []
        start = PathPlanner.start
        end = PathPlanner.end
        
        T:Tree = Tree(PathPlanner.start[0],PathPlanner.start[1])
        for _ in range(PathPlanner.N):
        
            # Select best node to extend (do while loop)

            while True:
                randomPos = ((end[0]+2)*random.random()-(end[0]+2)*random.random(),(end[1]+2)*random.random() - (end[1]+2)*random.random())
                nearestNode:TNode = T.getClosestNode(randomPos)
                nearestNodescore:float = math.exp(-point_2_point_distance(randomPos,end)/PathPlanner.k)

                rand_bound:float = random.random()
                
                if nearestNodescore > rand_bound and not(self.collision_detected((nearestNode.x,nearestNode.y),randomPos,lidar_segs)):
                    break
            
            
            # Extend tree with new node
            
            
            new_node = TNode(randomPos[0],randomPos[1])
            T.addNode(nearestNode,new_node)
            
            if point_2_point_distance(randomPos,(PathPlanner.end[0],PathPlanner.end[1])) < 0.5:
                PathPlanner.found:bool = True
                
                
                while new_node.x != start[0] and new_node.y != start[1]:
                    
                    new_pose = Pose()
                    new_pose.orientation.x = new_node.x
                    new_pose.orientation.y = new_node.y
                    new_pose.orientation.z = 0.0
                    self.poses.append(new_pose)
                    new_node = new_node.parent


                msg = PoseArray()
                msg.poses = self.poses
                print('sending path')
                publisher.publish(msg)
                break
            

    # @staticmethod
    # def collision_detected(node,point) -> bool:
        
    #     m = (node.y-point[0])/(node.x-point[1]) if (node.x-point[1]) != 0 else float('inf')
    #     c = point[1] - m*point[0]

    #     for segment in PathPlanner.lines:
            
    #         ix,iy = line_get_intersect([m,c],[segment.grad,segment.intersect])

    #         if (segment.min_x <= ix and ix <= segment.max_X) and (segment.min_y <= iy and iy <= segment.max_Y):
    #             logging.info("collision detected")
                
    #             return True
        
    #     return False
    

    def points_2_line(p1,p2):
        m = (p1[1]-p2[0])/(p1[0]-p2[1]) if (p1[0]-p2[1]) != 0 else float('inf')
        c = p2[1] - m*p2[0]
        return (m,c)
    
if __name__ == '__main__':
    rclpy.init()
    pp = PathPlanner()