from DataStructures import Tree, Node
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from feature_detection_py.feature_detection_py.LineDetector import SeedSegment

import logging
import random
import math
import matplotlib.pyplot as plt

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

    lines = []

    def __init__(self) -> None:
        super.__init__('minimal_subscriber')
        self.lines_subscription = self.create_subscription(Float32MultiArray,'line_segments',self.update_lines,10)
        super.__init__('minimal_publisher')

    def update_lines(msg):
        PathPlanner.lines = PathPlanner.get_seed_segs(msg)
        

    @staticmethod
    def get_seed_segs(line_data):
        seed_segs = []
        for i in range(0,len(line_data),6):
            seed_segs.append(SeedSegment.from_zipped_points(line_data[i],line_data[i+1],line_data[i+2],line_data[i+3],line_data[i+4],line_data[i+5]))


    @staticmethod
    def start(start: tuple[int,int],end: tuple[int,int])->Tree:
        '''
        RRT with euclidean distance heuristic
        '''
        T:Tree = Tree(start[0],start[1])
        for _ in range(PathPlanner.N):
        
            # Select best node to extend (do while loop)

            while True:
                randomPos = (end[0]*random.random(),end[1]*random.random())
                nearestNode:Node = T.getClosestNode(randomPos)
                nearestNodescore:float = math.exp(-point_2_point_distance(randomPos,end)/PathPlanner.k)
                rand_bound:float = random.random()
                
                if nearestNodescore > rand_bound:
                    break

            # Extend tree with new node
            if not(PathPlanner.collision_detected(nearestNode,randomPos)):
                T.addNode(nearestNode,Node(randomPos[0],randomPos[1]))
                if point_2_point_distance(randomPos,(end[0],end[1])) < 50:
                    PathPlanner.found:bool = True

        return T


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