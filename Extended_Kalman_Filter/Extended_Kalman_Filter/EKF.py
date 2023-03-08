
import numpy as np
import math
from seed_segment import SeedSegment
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from numpy.linalg import multi_dot
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
from scipy.interpolate import interp1d
import copy
import random
import matplotlib.pyplot as plt

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
class EKF(Node):
    
    C = 1
    system_state = np.array([0,0,0],'f')
    covariance_matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]],'f')
    prev_landmarks = []
    landmarks = []
    seed_segments = []


    def __init__(self) -> None:


        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-3,6)
        self.ax.set_ylim(-3,6)

        super().__init__("minimal_publisher")
        self.position_publisher = self.create_publisher(Pose,'/robot_position',10)

        super().__init__("minimal_subscriber")
        self.odom_sub = self.create_subscription(Odometry,"/odom",lambda msg :self.common_callback(msg),10)
        self.feature_sub = self.create_subscription(Float32MultiArray,"/line_segments",lambda msg : self.common_callback(msg),10)
        rclpy.spin(self)


    def common_callback(self,msg) -> None:
        if isinstance(msg,Odometry):
            self.odom_handler(msg,EKF.landmarks)
        if isinstance(msg,Float32MultiArray):
            EKF.update_landmarks(msg)


    def set_odometry(self,dx,dy,dtheta) -> None:
        EKF.system_state[0],EKF.system_state[1],EKF.system_state[2] = (dx,dy,dtheta)
    
    
    first = False
    @staticmethod
    def update_landmarks(line_data):
        if line_data.data == []:
            for s in EKF.seed_segments:
                s.reobserved = False
        else:
            noise = random.random()
            
            EKF.landmarks = [(1,1)]
            s = SeedSegment()
            s.x = 1
            s.y = 1
            s.reobserved = len(EKF.system_state) == 5
            if not(EKF.first):
                EKF.first = True
            
            
            EKF.seed_segments = [s]

            # s2 = SeedSegment()
            # s2.x = 2+noise
            # s2.y = 1+noise
            # if not(EKF.first):
            #     EKF.first = True
            # else:
            #     s2.reobserved = True
            


            # for i in range(0,len(line_data.data),7):
            #     s = SeedSegment.from_Float32MultiArray(line_data.data[i],line_data.data[i+1],line_data.data[i+2],line_data.data[i+3],line_data.data[i+4],line_data.data[i+5],bool(line_data.data[i+6]))
                
            #     s.x = s.x 
            #     s.y = s.y 
            #     EKF.landmarks.append((s.x,s.y))
            #     EKF.seed_segments.append(s)
            
    
    def odom_handler(self,odom:Odometry,landmarks:list):
        
        
        state_est,cov_est = self.make_prediction(EKF.system_state,EKF.covariance_matrix,odom)
        state_est,cov_est = self.update_prediction(state_est,cov_est,landmarks)

        EKF.system_state = state_est
        EKF.covariance_matrix = cov_est

        

        msg = Pose()
        msg.orientation.x = float(EKF.system_state[0])
        msg.orientation.y = float(EKF.system_state[1])
        msg.orientation.z = float(EKF.system_state[2])
        
        self.position_publisher.publish(msg)
        self.ax.plot(state_est[0],state_est[1],'r.')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


    def make_prediction(self,state_est:np.array,cov_est:np.array,odom:Odometry) -> list:
        '''
        Performs prediction step of EKF SLAM
        '''
        ST_SZ = len(state_est)
        dx,dy,_ = self.get_odom_change(odom,state_est)

        odom_pos:list = self.get_odom_pos(odom)
        state_est[0],state_est[1],state_est[2] = odom_pos[0],odom_pos[1],odom_pos[2]
        
        Fx = np.hstack((np.eye(3), np.zeros((3, 2 * int((len(state_est)-3)/2)))))
        jF = np.array([[0.,0.,-dy],[0.,0.,dx],[0.,0.,0.]],'f')
        
        est_jacobian = np.eye(len(state_est)) + Fx.transpose() @ jF @ Fx

        Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)])**2
       
        cov_est[0:ST_SZ,0:ST_SZ] = est_jacobian.transpose() @ cov_est[0:ST_SZ,0:ST_SZ] @ est_jacobian.transpose() + Fx.transpose() @ Cx @ Fx

        return state_est,cov_est
    
    def update_prediction(self,state_est,cov_est,landmarks) -> list:
        
        for lindex in range(len(landmarks)):
            
            minid = 3+2*lindex # points to x coord of current landmark
            
            if not(EKF.seed_segments[lindex].reobserved):   # add new landmark
                EKF.seed_segments[lindex].reobserved = True
                
                temp_state = np.hstack((state_est, np.array((landmarks[lindex]),'f')))
                temp_cov = np.pad(cov_est,[(0,2),(0,2)],mode='constant')
                state_est = temp_state
                cov_est = temp_cov
        

            lm = landmarks[lindex]
            
            y, S, H = self.get_innovation(lm,state_est,cov_est,lindex)
            
            K = (cov_est @ H.transpose()) @ np.linalg.inv(S)
            state_est = state_est + (K @ y)
            
            cov_est = (np.eye(len(state_est))-K @ H) @ cov_est

        return state_est,cov_est

    
    def get_innovation(self,lm,state_est,cov_est,minid)-> list:
        delta = lm - state_est[0:2]
        q = (delta.transpose() @ delta)  # TODO CHANGED
        zangle = math.atan2(delta[1],delta[0]) - state_est[2]
        z = np.array([math.sqrt(q),EKF.restrict_angle(zangle)],'f')

        zp = self.get_measurement_mat(state_est[minid],state_est)

        y = (z-zp).transpose()
        y[1] = self.restrict_angle(y[1])

        Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)])**2
        H = self.jacobianH(q,delta,state_est,minid+1)
        S = H @ cov_est @ H.transpose() + Cx[0:2,0:2]
        return y, S, H
        

    def jacobianH(self,q,delta,x,i) -> list:
        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0], - sq * delta[1], 0, sq * delta[0], sq * delta[1]],
                  [delta[1], - delta[0], - q, - delta[1], delta[0]]])

        G = G / q
        nLM = int((len(x)-3)/2)
        
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

        F = np.vstack((F1, F2))

        H = G @ F

        return H

    def get_measurement_mat(self,lm_pos,state_est):
        delta = lm_pos - state_est[0:2]
        q = (delta.transpose() @ delta)
        zangle = math.atan2(delta[1],delta[0]) - state_est[2]
        return np.array([math.sqrt(q),EKF.restrict_angle(zangle)],'f')

    @staticmethod
    def restrict_angle(ang) -> float:
        if ang > 2*math.pi:
            ang -= 2*math.pi
        if ang < 0:
            ang += 2*math.pi
        return ang

    @staticmethod
    def get_odom_change(odom:Odometry,state_est:np.array) -> list:
        x,y,theta = EKF.get_odom_pos(odom)
        return [x - state_est[0],y - state_est[1],theta - state_est[2]]

    @staticmethod
    def get_odom_pos(odom:Odometry) -> list:
        newPos = odom.pose.pose.position
        quat = odom.pose.pose.orientation
        w = quat.w
        x,y,z = (quat.x,quat.y,quat.z)
        new_theta = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) + math.pi 
        return [newPos.x,newPos.y,new_theta]
    

    def plot_covariance(self,cov_mat,centre_pos=(0,0)):
        w,_ = np.linalg.eig(cov_mat)
        asq = w[0]
        bsq = w[2]
        a = math.sqrt(asq)
        b = math.sqrt(bsq)
        self.ax.plot(centre_pos[0] ,centre_pos[1] ,'y*')
        xs = []
        ys = []
        i = 0
        while i < 4*math.pi + 0.1:
            r = a*b/(math.sqrt(b*math.cos(i)**2 + a*math.sin(i)**2))
            xs.append(centre_pos[0] + r*math.cos(i))
            ys.append(centre_pos[1] + r*math.sin(i))
            i += 0.01
        self.ax.plot(xs,ys,'m')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.ax.cla()

        

if __name__ == "__main__":
    rclpy.init()
    ekf = EKF()
