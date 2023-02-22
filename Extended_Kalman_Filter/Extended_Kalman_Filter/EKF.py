
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


class EKF(Node):

    interpolator = interp1d([-1,1],[0,2*math.pi])
    curr_landmarks = 0
    C = 1
    system_state = np.array([0,0,0])
    covariance_matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    landmarks = []

    def __init__(self) -> None:

        super().__init__("minimal_publisher")
        self.position_publisher = self.create_publisher(Pose,'/robot_position',10)

        super().__init__("minimal_subscriber")
        self.odom_sub = self.create_subscription(Odometry,"/odom",lambda msg :self.common_callback(msg),10)
        self.feature_sub = self.create_subscription(Float32MultiArray,"/line_segments",lambda msg : self.common_callback(msg),10)
        rclpy.spin(self)


    def common_callback(self,msg) -> None:
        if isinstance(msg,Odometry):
            EKF.extended_kalman_filter(msg,self.position_publisher)
        if isinstance(msg,Float32MultiArray):
            EKF.update_landmarks(msg)


    def set_odometry(self,dx,dy,dtheta) -> None:
        EKF.system_state[0],EKF.system_state[1],EKF.system_state[2] = (dx,dy,dtheta)


    @staticmethod
    def update_landmarks(line_data):
        for i in range(0,len(line_data.data),6):
            s = SeedSegment.from_Float32MultiArray(line_data.data[i],line_data.data[i+1],line_data.data[i+2],line_data.data[i+3],line_data.data[i+4],line_data.data[i+5])
            EKF.landmarks.append((s.x,s.y))


    @staticmethod
    def extended_kalman_filter(msg:Odometry,publisher):
        this_run_landmarks = EKF.landmarks.copy()
        
        '''
        step 1: update the current state using the odometry data
        '''
        
        newPos = msg.pose.pose.position
        new_theta = EKF.interpolator(msg.pose.pose.orientation.w)
        
        print(f'[{new_theta}]')
        dx,dy,dtheta = [EKF.system_state[0] - newPos.x,EKF.system_state[1] - newPos.y,EKF.system_state[2] - new_theta]

        EKF.system_state[0] = newPos.x
        EKF.system_state[1] = newPos.y
        EKF.system_state[2] = new_theta

        J_prediction = np.array([[1,0,-dy],[0,1,dx],[0,0,1]])
        # Q_noise = EKF.C*np.matmul(np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta]),np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta]))
        EKF.covariance_matrix[0:3,0:3] = multi_dot((J_prediction,EKF.covariance_matrix[0:3,0:3],J_prediction))  # TODO add Q_noise to this value
        
        '''
        step 2: update the state from reobserved landmarks
        '''
        
        for i in range(len(EKF.system_state)-3):
            
            cvWidth,cvHeight = EKF.covariance_matrix.shape

            # measurement model
            L_range = math.sqrt((this_run_landmarks[i][0] - EKF.system_state[0])**2 + (this_run_landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((this_run_landmarks[i][1] - EKF.system_state[1])/(this_run_landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
            

            A,B,C = [(EKF.system_state[0] - this_run_landmarks[i][0])/L_range,(EKF.system_state[1] - this_run_landmarks[i][1])/L_range,0]
            D,E,F = [(EKF.system_state[1] - this_run_landmarks[i][1])/L_range**2,(EKF.system_state[0] - this_run_landmarks[i][0])/L_range**2,-1]
            J_measurement = np.array([[0]*cvWidth,[0]*cvWidth])
            J_measurement[0,0],J_measurement[0,1],J_measurement[0,2] = A,B,C
            J_measurement[1,0],J_measurement[1,1],J_measurement[1,2] = D,E,F
            J_measurement[0,i] = -A
            J_measurement[0,i+1] = -B
            J_measurement[1,i] = -D
            J_measurement[1,i+1] = -E


            #J_measurement = np.array([[A,B,C] + [0]*2*(i)+  [-A,-B]+ [0]*(cvWidth-5-i) ,[D,E,F] + [0]*2*(i) + [-D,-E] + [0]*(cvWidth-5-i) ])
            
            
            R = np.array([[L_range,0],[0,L_bearing*2*math.pi/360]])
            var = multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()))+R
            K_gain = multi_dot((EKF.covariance_matrix,J_measurement.transpose(),np.linalg.inv(multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()) )+R)))

            np.add(EKF.system_state, K_gain)

        '''
        step 3: Update covariance matrix and system state to include new landmarks
        '''
        EKF.system_state = np.array(EKF.system_state[0:3])

        J_xr = np.array(([1,0,-dy],[0,1,dx]))
        J_z = np.array(([math.cos(EKF.system_state[2] + dtheta),math.sin(EKF.system_state[2] + dtheta)],[math.sin(EKF.system_state[2] + dtheta), math.cos(EKF.system_state[2] + dtheta)]))
        
        for i in range(len(this_run_landmarks)):
            
            L_range = math.sqrt((this_run_landmarks[i][0] - EKF.system_state[0])**2 + (this_run_landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((this_run_landmarks[i][1] - EKF.system_state[1])/(this_run_landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 

            R = multi_dot((np.identity(2),np.array([[L_range,0],[0,L_bearing*2*math.pi/360]]),np.identity(2)))
            
            P_Rn1 = np.dot(EKF.covariance_matrix[0:3,0:3],J_xr.transpose())
            
            P_n1n1 = multi_dot((J_xr,EKF.covariance_matrix[0:3,0:3],J_xr.transpose())) + multi_dot((J_z,R,J_z.transpose()))

            EKF.covariance_matrix = np.pad(EKF.covariance_matrix,[(0,2),(0,2)],mode='constant')
            
            matW,matH = EKF.covariance_matrix.shape

            for j_0 in range(0,len(this_run_landmarks)): # TODO Remember that this used to say old_landmarks and therefore could cause a problem but wait and see 

                j = j_0 +1 # this is a dummy var cause we need to iterate from 1 to the end of landmarks but this was quicker than adding j+1 to all the j's
                
                temp = EKF.covariance_matrix[j*3:j*3+2,0:2]
                M = np.dot(EKF.covariance_matrix[j*3:j*3+2,0:2],J_xr)
                EKF.covariance_matrix[j*3:(j*3+2),matH-3:matH] = M
                EKF.covariance_matrix[matW-3 :matW,j*3:j*3+2] = M.transpose()


            EKF.covariance_matrix[matH-2:matH,matW-2:matW] = P_n1n1
            EKF.covariance_matrix[matH-2:matH,0:3] = P_Rn1.transpose()
            EKF.covariance_matrix[0:3,matW-2:matW] = P_Rn1

            # add landmark to system state vector
            np.append(EKF.system_state,this_run_landmarks[i][0])
            np.append(EKF.system_state,this_run_landmarks[i][1])

        # send robots position to robot_pos topic
        msg = Pose()
        msg.orientation.x = float(EKF.system_state[0])
        msg.orientation.y = float(EKF.system_state[1])
        msg.orientation.z = float(EKF.system_state[2])
        print(EKF.system_state[0],EKF.system_state[1],EKF.system_state[2])
        publisher.publish(msg)
        EKF.curr_landmarks = len(this_run_landmarks)


if __name__ == "__main__":
    rclpy.init()
    ekf = EKF()
