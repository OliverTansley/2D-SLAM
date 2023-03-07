import math
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
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

    size = 15


    def __init__(self,m=0,c=0,xpnts=[],ypnts=[]) -> None:

        self.grad = m
        self.intersect = c
        self.max_X = max(xpnts)
        self.max_Y = max(ypnts)
        self.min_x = min(xpnts)
        self.min_y = min(ypnts)
        self.points = list(zip(xpnts,ypnts))
        self.xpoints = xpnts
        self.ypoints = ypnts
        self.x = (self.min_x + self.max_X)/2
        self.y = (self.min_y + self.max_Y)/2
        self.reobserved_times = 0
        self.reobserved = False


    @classmethod
    def from_Float32MultiArray(cls,m,c,max_X,max_Y,min_x,min_y,reob,reTime):
        seed_seg = SeedSegment()
        seed_seg.grad = m
        seed_seg.intersect = c
        seed_seg.max_X = max_X
        seed_seg.max_Y = max_Y
        seed_seg.min_y = min_y
        seed_seg.min_x = min_x
        seed_seg.x = (seed_seg.min_x + seed_seg.max_X)/2
        seed_seg.y = (seed_seg.min_y + seed_seg.max_Y)/2
        seed_seg.reobserved = reob
        seed_seg.reobserved_times = reTime
        return seed_seg
    

    def show(self,ax) -> None:

        xs = [self.min_x,self.max_X]
        ys = []
        for x in xs:
            y = self.grad*x + self.intersect
            if y > self.max_Y:
                y = self.max_Y
            if y < self.min_y:
                y = self.min_y
            ys.append(y)

       
        ax.plot(xs,ys,"b")
        


class LineDetector(Node):

    
    seed_segments = []
    epsilon = 0.1
    sigma = 0.02
    Pmin = 10

    def __init__(self):
        
        self.robot_pos = None

        super().__init__("minimal_publisher")
        self.publisher= self.create_publisher(Float32MultiArray, '/line_segments', 10)    

        super().__init__("minimal_subscriber")
        self.subscription = self.create_subscription(LaserScan,"/scan",lambda msg : self.common_callback(msg),10)
        self.position_subscription = self.create_subscription(Pose,"/robot_position",lambda msg : self.common_callback(msg),10)

        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-12,12)
        self.ax.set_ylim(-12,12)
        rclpy.spin(self)


    def common_callback(self,msg):
        if isinstance(msg,LaserScan) and self.robot_pos != None:
            self.make_seed_segments(msg,self.publisher)
        if isinstance(msg,Pose):
            self.ax.plot(msg.orientation.x,msg.orientation.y,'k.')
            self.robot_pos = (msg.orientation.x,msg.orientation.y,msg.orientation.z)


    def make_seed_segments(self,lidar_msg,publisher) -> None:
        '''
        Adds seed segments to seed segment array
        '''

        for s in LineDetector.seed_segments:
            s.reobserved = False

        xs,ys = self.lidar_2_points(lidar_msg)

        valid_segment = True
        i = 0
        
        while i < len(xs):
            j = i + SeedSegment.size
            m,c = LineDetector.total_least_squares(xs[i:j],ys[i:j])

            for point_index in range(i,min(j,len(xs))):
                valid_segment=True
                
                if self.predicted_point_distance([m,c],[xs[point_index],ys[point_index]],lidar_msg,i) > LineDetector.sigma:
                    valid_segment = False    
                    break

                if point_2_line_distance([m,c],[xs[point_index],ys[point_index]]) > LineDetector.epsilon:
                    valid_segment = False
                    break
                
            if valid_segment:
                
                # seed segment region growing
                
                P_start = i-1
                P_end = j+1
                
                while P_end+1 < len(xs) and point_2_line_distance((m,c),(xs[P_end+1],ys[P_end+1])) < LineDetector.epsilon:
                    P_end += 1
                    m , c = LineDetector.total_least_squares(xs[i:P_end],ys[i:P_end])

                P_end -= 1

                while P_start > -1 and point_2_line_distance((m,c),(xs[P_start-1],ys[P_start-1])) < LineDetector.epsilon:
                    P_start -= 1
                    m , c = LineDetector.total_least_squares(xs[P_start:P_end],ys[P_start:P_end])
                
                P_start += 1


                new_segment = SeedSegment(m,c,xs[P_start:P_end],ys[P_start:P_end])
                
                if not(self.re_observed(new_segment)):
                    LineDetector.seed_segments.append(new_segment)

                msg = Float32MultiArray()
            
                features = []
                for s in LineDetector.seed_segments:
                    features.append(s.grad)
                    features.append(s.intersect)
                    features.append(s.max_X)
                    features.append(s.max_Y)
                    features.append(s.min_x)
                    features.append(s.min_y)
                    features.append(float(s.reobserved))

                msg.data = features
                publisher.publish(msg)

                i = P_end
            else:
                i += 1
                
        if xs == []:
            msg = Float32MultiArray()
            msg.data = []
            publisher.publish(msg)
        

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        

    def re_observed(self,new_segment):
        old_segment = self.get_closest_segment(new_segment)
        
        if old_segment == None or point_2_point_distance((old_segment.x,old_segment.y),(new_segment.x,new_segment.y)) > 0.7:
            return False

        accepted_points = 0
        for p in new_segment.points:
            if point_2_line_distance((old_segment.grad,old_segment.intersect),p) < 0.7:
                accepted_points += 1
        
        if accepted_points > len(new_segment.points)/4:
            old_segment.reobserved = True
            old_segment.max_X = new_segment.max_X
            old_segment.max_Y = new_segment.max_Y
            old_segment.min_x = new_segment.min_x
            old_segment.min_y = new_segment.min_y
            return True
            
        return False



    def get_closest_segment(self,new_segment):
        distance = float('inf')
        
        best_segment = None
        for old_segment in LineDetector.seed_segments:
            
            if point_2_point_distance((old_segment.x,old_segment.y),(new_segment.x,new_segment.y)) < distance:
                distance = point_2_point_distance((old_segment.x,old_segment.y),(new_segment.x,new_segment.y))
                best_segment = old_segment
        
        return best_segment



    def predicted_point_distance(self,seedline,point,laserScan:LaserScan,index) -> float:
        '''
        determines where the line, between a point and the robot intersects a given seed segment line
        '''
        if (point[0] - self.robot_pos[0]) == 0:
            pointline_grad = (point[1]-self.robot_pos[1])*float('inf')
        else:
            pointline_grad = (point[1]-self.robot_pos[1])/(point[0] - self.robot_pos[0]) 
        pointline_intercept = point[1] - pointline_grad*point[0]
        
        i_x = (seedline[1] - pointline_intercept)/(pointline_grad - seedline[0])
        i_y = (seedline[0]*pointline_intercept - seedline[1]*pointline_grad)/(seedline[0] - pointline_grad)
       
        return math.sqrt((point[0] - i_x)**2 + (point[1] - i_y)**2)


    # HELPER FUNCTIONS
    

    def lidar_2_points(self,lidar_msg):
        '''
        Converts raw lidar data to arrays of x and y coordinates relative to scanner
        '''
        lidar_ranges = lidar_msg.ranges
        ang_inc = lidar_msg.angle_increment
        xs = []
        ys = []
        
        noise = random.random()/5

        for s in LineDetector.seed_segments:
            
            s.show(self.ax)


        for measurement in range(0,len(lidar_ranges)):
            if lidar_ranges[measurement] != float('inf') :
                xs.append(self.robot_pos[0] - lidar_ranges[measurement] * math.sin(measurement*ang_inc + math.pi/2 + self.robot_pos[2]) ) 
                ys.append(self.robot_pos[1] + lidar_ranges[measurement] * math.cos(measurement*ang_inc + math.pi/2 + self.robot_pos[2]) )

        # self.ax.plot(xs,ys,'b.')

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
