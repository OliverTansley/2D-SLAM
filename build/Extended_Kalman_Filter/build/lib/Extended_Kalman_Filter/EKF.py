
import numpy as np
import math
from seed_segment import SeedSegment
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from numpy.linalg import multi_dot
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

class EKF(Node):

    C = 1
    system_state = np.array([0,0,0])
    covariance_matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    landmarks = [(1,1),(2,2)]

    def __init__(self) -> None:

        super().__init__("minimal_publisher")
        self.position_publisher = self.create_publisher(Pose2D,'robot_position',10)

        super().__init__("minimal_subscriber")
        self.odom_sub = self.create_subscription(Odometry,"/odom",lambda msg :EKF.extended_kalman_filter(msg,self.position_publisher),10)
        self.feature_sub = self.create_subscription(Float32MultiArray,"/line_segments",EKF.update_landmarks,10)
        rclpy.spin(self)


    def set_odometry(self,dx,dy,dtheta) -> None:
        EKF.system_state[0],EKF.system_state[1],EKF.system_state[2] = (dx,dy,dtheta)


    @staticmethod
    def update_landmarks(line_data):
        for i in range(0,len(line_data),6):
            s = SeedSegment.from_zipped_points(line_data[i],line_data[i+1],line_data[i+2],line_data[i+3],line_data[i+4],line_data[i+5])
            EKF.landmarks.append((s.x,s.y))

    @staticmethod
    def extended_kalman_filter(msg,publisher):

        # step 1: update the current state using the odometry data
        
        newPos = msg.pose.pose.orientation
        dx,dy,dtheta = [EKF.system_state[0] - newPos.x,EKF.system_state[1] - newPos.y,EKF.system_state[2] - newPos.w]
        EKF.system_state = (EKF.system_state[0] + newPos.x,EKF.system_state[1] + newPos.y,EKF.system_state[2] + newPos.w)
        
        J_prediction = np.array([[1,0,-dy],[0,1,dx],[0,0,1]])
         # Q_noise = EKF.C*np.matmul(np.array([dt*math.cos(theta),dt*math.sin(theta),dtheta]),np.array([dt*math.cos(theta),dt*math.sin(theta),dtheta]))
        EKF.covariance_matrix[0:3,0:3] = multi_dot((J_prediction,EKF.covariance_matrix[0:3,0:3],J_prediction)) 
        

        # step 2: update the state from reobserved landmarks

        for i in range(len(EKF.landmarks)-1):
            
            # measurement model
            L_range = math.sqrt((EKF.landmarks[i][0] - EKF.system_state[0])**2 + (EKF.landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((EKF.landmarks[i][1] - EKF.system_state[1])/(EKF.landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
            measurement_model = np.array([L_range,L_bearing])

            A,B,C = [(EKF.system_state[0] - EKF.landmarks[i][0])/L_range,(EKF.system_state[1] - EKF.landmarks[i][1])/L_range,0]
            D,E,F = [(EKF.system_state[1] - EKF.landmarks[i][1])/L_range**2,(EKF.system_state[0] - EKF.landmarks[i][0])/L_range**2,-1]
            J_measurement = np.array([[A,B,C] + [0]*2*(i)+  [-A,-B],[D,E,F] + [0]*2*(i) + [-D,-E]])

            # var = multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()))
            
            R = multi_dot((np.identity(2),np.array([[L_range,0],[0,L_bearing*2*math.pi/360]]),np.identity(2)))
            K_gain = multi_dot((EKF.covariance_matrix,J_measurement.transpose(),np.linalg.inv(multi_dot((J_measurement,EKF.covariance_matrix,J_measurement.transpose()) + R))))

            np.add(EKF.system_state, K_gain)

        # step 3: Update covariance matrix and system state to include new landmarks
        
        EKF.system_state = np.array(EKF.system_state[0:3])

        J_xr = np.array(([1,0,-dy],[0,1,dx]))
        J_z = np.array(([math.cos(EKF.system_state[2] + dtheta),math.sin(EKF.system_state[2] + dtheta)],[math.sin(EKF.system_state[2] + dtheta), math.cos(EKF.system_state[2] + dtheta)]))
        EKF.landmarks = [(1,1),(2,1)]
        for i in range(len(EKF.landmarks) -1):
            
            L_range = math.sqrt((EKF.landmarks[i][0] - EKF.system_state[0])**2 + (EKF.landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((EKF.landmarks[i][1] - EKF.system_state[1])/(EKF.landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 

            R = multi_dot((np.identity(2),np.array([[L_range,0],[0,L_bearing*2*math.pi/360]]),np.identity(2)))
            P_Rn1 = np.dot(EKF.covariance_matrix[0:3][0:3],J_xr.transpose())
            P_n1n1 = multi_dot((J_xr,EKF.covariance_matrix,J_xr.transpose())) + multi_dot((J_z,R,J_z.transpose()))

            EKF.covariance_matrix = np.pad(EKF.covariance_matrix,[(0,3),(0,3)],mode='constant')
            
            matW,matH = EKF.covariance_matrix.shape

            for j_0 in range(0,len(EKF.landmarks)): # TODO Remember that this used to say old_landmarks and therefore could cause a problem but wait and see 

                j = j_0 +1 # this is a dummy var cause we need to iterate from 1 to the end of landmarks but this was quicker than adding j+1 to all the j's
                M = np.dot(J_xr,EKF.covariance_matrix[j*3:(j+1)*3][0:3])
                EKF.covariance_matrix[j*3:(j+1)*3][matW-3 -1:matW-1] = M
                EKF.covariance_matrix[matH-3 -1:matH-1][j*3:(j+1)*3] = M.transpose()

            EKF.covariance_matrix[matH-3 -1:matH-1][matW-3 -1:matW-1] = P_n1n1
            EKF.covariance_matrix[matH-3 -1:matH-1][0:3] = P_Rn1.transpose()
            EKF.covariance_matrix[0:3][matW-3 -1:matW-1] = P_Rn1

            # add landmark to system state vector
            np.append(EKF.system_state,EKF.landmarks[i][0])
            np.append(EKF.system_state,EKF.landmarks[i][1])

        # send robots position to robot_pos topic
        msg = Pose2D()
        msg.x = EKF.system_state[0]
        msg.y = EKF.system_state[1]
        msg.theta = EKF.system_state[2]
        publisher.publish(msg)


 # def get_measurement_model():
    #     L_range = math.sqrt((landmark.x - x)**2 + (lambda_y - y)**2) + v_r
    #     L_bearing = math.atan((lambda_y-y)/(landmark.x-x)) - theta + v_theta
    #     return np.array([L_range,L_bearing])


    # def get_prediction_model(self):
    #     row1 = self.system_state[0] + dt*math.cos(self.system_state[2]) + q*dt*math.cos(self.system_state[2])
    #     row2 = y + dt*math.sin(theta) + q*dt*math.sin(theta)
    #     row3 = theta + dtheta + q*dtheta
    #     return np.array([row1,row2,row3])

if __name__ == "__main__":
    rclpy.init()
    ekf = EKF()