
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
class EKF(Node):
    
    C = 0.01
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
            self.extended_kalman_filter(msg,self.position_publisher)
        if isinstance(msg,Float32MultiArray):
            EKF.update_landmarks(msg)


    def set_odometry(self,dx,dy,dtheta) -> None:
        EKF.system_state[0],EKF.system_state[1],EKF.system_state[2] = (dx,dy,dtheta)
    
    
    @staticmethod
    def update_landmarks(line_data):
        
        EKF.landmarks = []
        EKF.seed_segments = []
        for i in range(0,len(line_data.data),7):
            s = SeedSegment.from_Float32MultiArray(line_data.data[i],line_data.data[i+1],line_data.data[i+2],line_data.data[i+3],line_data.data[i+4],line_data.data[i+5],bool(line_data.data[i+6]))
            s.x = s.x 
            s.y = s.y 
            EKF.landmarks.append((s.x,s.y))
            EKF.seed_segments.append(s)

    

    def extended_kalman_filter(self,msg:Odometry,publisher):

        this_run_landmarks = EKF.prev_landmarks
        this_run_new_landmarks = EKF.landmarks
        # self.ax.plot([msg.pose.pose.position.x ],[msg.pose.pose.position.y])
        
        # self.ax.plot(EKF.system_state[0],EKF.system_state[1],'r.')

        '''
        step 1: update the current state using the odometry data
        '''
        
        newPos = msg.pose.pose.position 
        
        quat = msg.pose.pose.orientation
        w = quat.w
        x = quat.x
        y = quat.y
        z = quat.z
        new_theta = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) + math.pi 
        
        dx = newPos.x - EKF.system_state[0] 
        dy = newPos.y - EKF.system_state[1] 
        dtheta = new_theta - EKF.system_state[2]

        # update state vector to contain new estimates of the robots position (purely odometry)
        EKF.system_state[0] = newPos.x 
        EKF.system_state[1] = newPos.y 
        EKF.system_state[2] = new_theta 

        old = copy.deepcopy(EKF.system_state)

        dt = 1  #TODO THIS IS WHY COV MATRIX TOO BIG

        # update the jacobian (the derivative of the state vector)
        J_prediction = np.array([[1,0,-dy],[0,1,dx],[0,0,1]],'f')
        Q_noise = EKF.C*np.matmul(np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta],'f'),np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta],'f'))
        
        # update the estimate of the covariance of the robots Pose
        EKF.covariance_matrix[0:3,0:3] = multi_dot((J_prediction.transpose(),EKF.covariance_matrix[0:3,0:3],J_prediction)) + Q_noise 
        
        # update the first three rows of the covariance matrix
        EKF.covariance_matrix[0:3,:] = np.dot(J_prediction,EKF.covariance_matrix[0:3,:])

        '''
        step 2: update the state from reobserved landmarks
        '''
        for i in range(math.ceil((len(EKF.system_state)-3)/2)):
            
            if EKF.seed_segments[i].reobserved:
                
                cvWidth, _ = EKF.covariance_matrix.shape
                
                # get the range and bearing of the reobserved landmarks old observation
                L_range = math.sqrt((EKF.system_state[i+3+0] - EKF.system_state[0])**2 + (EKF.system_state[i+3+1] - EKF.system_state[1])**2) 
                L_bearing = math.atan((EKF.system_state[i+3+1] - EKF.system_state[1])/(EKF.system_state[i+3+0]-EKF.system_state[0])) - EKF.system_state[2] 

                # old landmark measurement matrix
                # h = self.get_measurement_matrix(L_range,L_bearing);

                # old landmark measurement matrix
                # h = np.array([L_range,L_bearing],'f')
                delta = np.array([this_run_landmarks[i][0] - (EKF.system_state[i+3+0]+dx) ,this_run_landmarks[i][1]- (EKF.system_state[i+3+1]+dy) ])
                
                # create jacobian of the measurement model
                A = (EKF.system_state[0] - this_run_landmarks[i][0])/L_range
                B = (EKF.system_state[1] - this_run_landmarks[i][1])/L_range
                C = 0
                D = (EKF.system_state[1] - this_run_landmarks[i][1])/L_range**2
                E = (EKF.system_state[0] - this_run_landmarks[i][0])/L_range**2
                F = -1
                J_measurement = np.array([[0]*cvWidth,[0]*cvWidth],'f')
                J_measurement[0][0] = A
                J_measurement[0][1] = B
                J_measurement[0][2] = C
                J_measurement[1][0] = D
                J_measurement[1][1] = E
                J_measurement[1][2] = F
                J_measurement[0][3+2*i] = -A
                J_measurement[0][3+2*i+1] = -B
                J_measurement[1][3+2*i] = -D
                J_measurement[1][3+2*i+1] = -E
                
                # measurement model for new measurement of the landmark
                L_range = math.sqrt((this_run_landmarks[i][0] - EKF.system_state[0])**2 + (this_run_landmarks[i][1] - EKF.system_state[1])**2) 
                L_bearing = math.atan((this_run_landmarks[i][1] - EKF.system_state[1])/(this_run_landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
                Z = np.array([L_range,L_bearing],'f')

                # print(f'({this_run_landmarks[i][0]},{this_run_landmarks[i][1]}) ({this_run_new_landmarks[i][0]},{this_run_new_landmarks[i][1]})')

                # calculate kalman gain for this landmark
                # R = np.array([[1/100,0],[0,2*math.pi/360]],'f') # TODO ALSO MIGHT MAKE SYSTEM VALUES EXPLODE
                K_gain = multi_dot((EKF.covariance_matrix,J_measurement.transpose(),np.linalg.inv(multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()) ))))
            
                EKF.system_state += np.dot(K_gain , (delta))
                EKF.covariance_matrix -= multi_dot((K_gain,np.linalg.inv(multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()) )),K_gain.transpose())) 
        
        '''
        step 3: Update covariance matrix and system state to include new landmarks
        '''
        
        J_xr = np.array(([1,0,-dy],[0,1,dx]),'f')
        J_z = np.array(([math.cos(EKF.system_state[2] + dtheta),math.sin(EKF.system_state[2] + dtheta)],[math.sin(EKF.system_state[2] + dtheta), math.cos(EKF.system_state[2] + dtheta)]),'f')
        
        for i in range(len(this_run_landmarks),len(this_run_new_landmarks)):
            
            matW,matH = EKF.covariance_matrix.shape

            # add landmark to system state vector
            EKF.system_state = np.append(EKF.system_state,this_run_new_landmarks[i][0])
            EKF.system_state = np.append(EKF.system_state,this_run_new_landmarks[i][1])

            # calculate R matrix for current landmark
            L_range = math.sqrt((this_run_new_landmarks[i][0] - EKF.system_state[0])**2 + (this_run_new_landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((this_run_new_landmarks[i][1] - EKF.system_state[1])/(this_run_new_landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
            R = multi_dot((np.identity(2),np.array([[L_range,0],[0,L_bearing*2*math.pi/360]],'f'),np.identity(2)))

            # extend covariance matrix and add the new landmarks covariance
            EKF.covariance_matrix = np.pad(EKF.covariance_matrix,[(0,2),(0,2)],mode='constant')
            P_n1n1 = multi_dot((J_xr,EKF.covariance_matrix[0:3,0:3],J_xr.transpose())) + multi_dot((J_z,R,J_z.transpose()))
            EKF.covariance_matrix[matH-2:matH,matW-2:matW] = P_n1n1
            
            # add the robot - landmark covariance to the covariance matrix
            P_Rn1 = np.dot(EKF.covariance_matrix[0:3,0:3],J_xr.transpose())
            EKF.covariance_matrix[0:3,matW-2:matW] = P_Rn1

            # add landmark robot covariance to covariance matrix
            EKF.covariance_matrix[matH-2:matH,0:3] = P_Rn1.transpose()

            # add the landmark - landmark covariances to the remaining empty spots in the matrix
            
            for j in range(1,len(this_run_landmarks)+1):  
                
                M = np.dot(EKF.covariance_matrix[j*3:j*3+2,0:2],J_xr)
                EKF.covariance_matrix[j*3:(j*3+2),matH-3:matH] = M
                EKF.covariance_matrix[matW-3 :matW,j*3:j*3+2] = M.transpose()

        # set all the new landmarks as previous landmarks for the next time step
        EKF.prev_landmarks = np.array(copy.deepcopy(EKF.landmarks),'f')
        new = copy.deepcopy(EKF.system_state)
        try:
            print(new - old)
        except:
            pass
        print("--")
        # send robots position to robot_pos topic
        msg = Pose()
        msg.orientation.x = float(EKF.system_state[0])
        msg.orientation.y = float(EKF.system_state[1])
        msg.orientation.z = float(EKF.system_state[2])
        
    
       
        # print(np.array2string(EKF.system_state).replace('[[',' [').replace(']]',']'))
        # print(np.array2string(EKF.covariance_matrix).replace('[[',' [').replace(']]',']'))
        

        publisher.publish(msg)
        self.ax.plot(EKF.system_state[0],EKF.system_state[1],'.')
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        EKF.prev_msg = msg

if __name__ == "__main__":
    rclpy.init()
    ekf = EKF()
