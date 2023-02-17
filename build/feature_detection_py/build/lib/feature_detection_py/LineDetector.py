import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


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


def point_2_line_distance(line,point):
    '''
    determines a points euclidean distance from the seed segment line
    '''
    
    perpendicular_grad = -1/line[0]
    perpendicular_intercept = point[1] - perpendicular_grad*point[0]

    i_x = (perpendicular_intercept - line[1])/(line[0] - perpendicular_grad)
    i_y = (line[1]*perpendicular_grad - line[0]*perpendicular_intercept)/(perpendicular_grad - line[0])

    return math.sqrt((point[0] - i_x)**2 + (point[1] - i_y)**2)


class SeedSegment:

    size = 30

    def __init__(self,m,c,xpnts,ypnts) -> None:

        self.grad = m
        self.intersect = c
        self.max_X = max(xpnts)
        self.max_Y = max(ypnts)
        self.min_x = min(xpnts)
        self.min_y = min(ypnts)
        self.points = list(zip(xpnts,ypnts))
        self.x = (self.min_x + self.max_X)/2
        self.y = (self.min_y + self.max_Y)/2

    def plot_line(self) -> None:

        xs = [self.min_x,self.max_X]
        ys = []
        for x in xs:
            y = self.grad*x + self.intersect
            if y > self.max_Y:
                y = self.max_Y
            if y < self.min_y:
                y = self.min_y
            ys.append(y)


        plt.plot(xs,ys,"r")


class LineDetector(Node):


    seed_segments = []
    epsilon = 0.03
    sigma = 0.03
    Pmin = 30

    def __init__(self):
        
        
        self.publisher_ = self.create_publisher(Float32MultiArray, 'line_segments', 10)     # CHANGE
        
        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(LaserScan,"/scan",LineDetector.make_seed_segments,10)
        rclpy.create_node("LineDetector")
        rclpy.spin(self)



    @staticmethod
    def make_seed_segments(lidar_data) -> None:
        '''
        Adds seed segments to seed segment array
        '''
        
        lidar_data = lidar_data.ranges

        xs,ys = LineDetector.lidar_2_points(lidar_data)
        
        plt.plot(xs,ys,"g.")
        valid_segment = True
        i = 0
        while i < len(lidar_data):
            j = i + SeedSegment.size
            m,c = LineDetector.total_least_squares(xs[i:j],ys[i:j])
            
            
            for point_index in range(i,min(j,len(lidar_data))):
                valid_segment=True
                
                if LineDetector.predicted_point_distance([m,c],[xs[point_index],ys[point_index]]) > LineDetector.sigma:
                    valid_segment = False    
                    break

                if point_2_line_distance([m,c],[xs[point_index],ys[point_index]]) > LineDetector.epsilon:
                    valid_segment = False
                    break
                
            if valid_segment:
                
                # seed segment region growing
                
                P_start = i-1
                P_end = j+1
                
                while P_end < len(lidar_data) and point_2_line_distance((m,c),(xs[j+1],ys[j+1])) < 0.002:
                    m , c = LineDetector.total_least_squares(xs[i:P_end],ys[i:P_end])
                    P_end += 1

                P_end -= 1

                while P_start > -1 and point_2_line_distance((m,c),(xs[P_start],ys[P_start])) < 0.002:
                    m , c = LineDetector.total_least_squares(xs[P_start:P_end],ys[P_start:P_end])
                    P_start -= 1
                
                P_start += 1

                new_segment = SeedSegment(m,c,xs[P_start:P_end],ys[P_start:P_end])
                LineDetector.seed_segments.append(new_segment)
                i = P_end
            else:
                i += 1
                
    def predicted_point_distance(seedline,point) -> float:
        '''
        determines where the line, between a point and the robot intersects a given seed segment line
        '''

        pointline_grad = point[1]/(point[0] +0.00000001) # TODO remove static assumption
        pointline_intercept = 0 # intercept is always 0 as robot is still located at (0,0)
                                # this will be changed once the static assumption is removed
        
        i_x = (seedline[1] - pointline_intercept)/(pointline_grad - seedline[0])
        i_y = (seedline[0]*pointline_intercept - seedline[1]*pointline_grad)/(seedline[0] - pointline_grad)
       
        return math.sqrt((point[0] - i_x)**2 + (point[1] - i_y)**2)


    # HELPER FUNCTIONS
    

    def lidar_2_points(lidar_ranges) -> List[List[float]]:
        '''
        Converts raw lidar data to arrays of x and y coordinates relative to scanner
        '''
        xs = []
        ys = []
        for measurement in range(0,len(lidar_ranges)):
            xs.append(lidar_ranges[measurement] * math.sin(math.radians(measurement)))
            ys.append(lidar_ranges[measurement] * math.cos(math.radians(measurement)))

        return xs,ys


    def points_2_line(Xvals,Yvals) -> List[float]:
        '''
        Returns gradient and intercept of least squares regression line of points provided
        '''
        Xpoints = np.array(Xvals)
        Ypoints = np.array(Yvals)
        Xpoints = Xpoints[np.isfinite(Xpoints)]
        Ypoints = Ypoints[np.isfinite(Ypoints)]
        
        return (np.linalg.pinv(np.column_stack((Xpoints,np.ones(Xpoints.size))))) @ Ypoints


    def total_least_squares(Xvals,Yvals) -> List[float]:
        
        def f(B, x):
            '''Linear function y = m*x + b'''
            return B[0]*x + B[1]

        Xvals = np.array(Xvals)
        Yvals = np.array(Yvals)
        linear = Model(f)
        mydata = RealData(Xvals, Yvals)
        myodr = ODR(mydata, linear, beta0=[1., 2.])
        myoutput = myodr.run()

        return myoutput.beta

if __name__=="__main__":
    rclpy.init()
    ld = LineDetector()