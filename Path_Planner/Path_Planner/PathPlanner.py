from DataStructures.Tree import Tree, Node

from Utils.ArithmeticUtil import *

import LineDetection.LineDetection as LD
import logging
import random
import math
import matplotlib.pyplot as plt


class PathPlanner:

    N:int = 40
    found:bool = False
    k:int = 200


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

        for segment in LD.LineDetector.seed_segments:
            
            ix,iy = line_get_intersect([m,c],[segment.grad,segment.intersect])

            if (segment.min_x <= ix and ix <= segment.max_X) and (segment.min_y <= iy and iy <= segment.max_Y):
                logging.info("collision detected")
                
                return True
        
        return False