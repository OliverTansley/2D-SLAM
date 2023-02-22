from DataStructures import Tree, TNode
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from seed_segment import SeedSegment
from geometry_msgs.msg import Pose, PoseArray
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

    start = (0,0)
    end = (10,10)

    lines = []

    def __init__(self) -> None:

        super().__init__('minimal_publisher')
        self.path_publisher = self.create_publisher(PoseArray,'/goal_positions',10)
        

        super().__init__('minimal_subscriber')
        self.lines_subscription = self.create_subscription(Float32MultiArray,'/line_segments',lambda msg :self.common_callback(msg),10)
        self.position_subscription = self.create_subscription(Pose,'/robot_position',lambda msg :self.common_callback(msg),10)
        PathPlanner.get_path(self.path_publisher)
        
        rclpy.spin(self)


    def update_lines(msg):
        PathPlanner.lines = PathPlanner.get_seed_segs(msg)
        
    def update_pos(msg):
        PathPlanner.start = (msg.orientation.x,msg.orientation.y)


    def common_callback(self,msg):
        if isinstance(msg,Float32MultiArray):
            self.update_lines(msg)
        if isinstance(msg,Pose):
            PathPlanner.update_pos(msg)


    @staticmethod
    def get_seed_segs(line_data):
        seed_segs = []
        for i in range(0,len(line_data),6):
            seed_segs.append(SeedSegment.from_zipped_points(line_data[i],line_data[i+1],line_data[i+2],line_data[i+3],line_data[i+4],line_data[i+5]))
        return seed_segs


    @staticmethod
    def get_path(publisher)->None:
        '''
        RRT with euclidean distance heuristic
        '''
        start = PathPlanner.start
        end = PathPlanner.end
        print(PathPlanner.lines)
        T:Tree = Tree(PathPlanner.start[0],PathPlanner.start[1])
        for _ in range(PathPlanner.N):
        
            # Select best node to extend (do while loop)

            while True:
                randomPos = ((end[0]+2)*random.random(),(end[1]+2)*random.random())
                nearestNode:TNode = T.getClosestNode(randomPos)
                nearestNodescore:float = math.exp(-point_2_point_distance(randomPos,end)/PathPlanner.k)
                rand_bound:float = random.random()
                
                if nearestNodescore > rand_bound:
                    break

            # Extend tree with new node
            if not(PathPlanner.collision_detected(nearestNode,randomPos)):
                new_node = TNode(randomPos[0],randomPos[1])
                T.addNode(nearestNode,new_node)
                if point_2_point_distance(randomPos,(PathPlanner.end[0],PathPlanner.end[1])) < 5:
                    PathPlanner.found:bool = True
                    
                    poses = []
                    while new_node.x != start[0] and new_node.y != start[1]:
                        new_pose = Pose()
                        new_pose.orientation.x = new_node.x
                        new_pose.orientation.y = new_node.y
                        new_pose.orientation.z = 0.0
                        poses.append(new_pose)
                        new_node = new_node.parent

                    poses.reverse()

                    msg = PoseArray()
                    msg.poses = poses
                    print('sending path')
                    publisher.publish(msg)
                    break

    @staticmethod
    def collision_detected(node,point) -> bool:
        
        m = (node.y-point[0])/(node.x-point[1]) if (node.x-point[1]) != 0 else float('inf')
        c = point[1] - m*point[0]

        for segment in PathPlanner.lines:
            
            ix,iy = line_get_intersect([m,c],[segment.grad,segment.intersect])

            if (segment.min_x <= ix and ix <= segment.max_X) and (segment.min_y <= iy and iy <= segment.max_Y):
                logging.info("collision detected")
                
                return True
        
        return False
    
if __name__ == '__main__':
    rclpy.init()
    pp = PathPlanner()