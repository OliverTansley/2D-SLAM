def extended_kalman_filter(self,msg:Odometry,publisher):
        
        EKF.landmarks = EKF.landmarks
        
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

        # self.ax.plot(EKF.system_state[0],EKF.system_state[1],'r.')

        dt = 0.001

        # update the jacobian (the derivative of the state vector)
        jacobian_control = np.array([[1,0,-dy],[0,1,dx],[0,0,1]],'f')
        Q_noise = EKF.C*np.matmul(np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta],'f'),np.array([dt*math.cos(EKF.system_state[2]),dt*math.sin(EKF.system_state[2]),dtheta],'f'))
        
        # update the estimate of the covariance of the robots Pose
        EKF.covariance_matrix[0:3,0:3] = multi_dot((jacobian_control,EKF.covariance_matrix[0:3,0:3],jacobian_control)) +Q_noise 
        
        

        self.plot_covariance(EKF.covariance_matrix[0:3,0:3],(EKF.system_state[0],EKF.system_state[1]))

        '''
        step 2: update the state from reobserved landmarks
        '''
        for i in range(math.floor((len(EKF.system_state)-3)/2)):
            
            if EKF.seed_segments[i].reobserved:
                print("=====")
                print(f'landmark number: {i}')
                print(f'x coord        : {EKF.system_state[2*i+3+0]}')
                print(f'y coord        : {EKF.system_state[2*i+3+1]}')
                for j in range(len(EKF.seed_segments)):
                    self.ax.plot(EKF.system_state[2*i+3+0],EKF.system_state[2*i+3+1],'k+')
                    print(f'{EKF.seed_segments[j].x,EKF.seed_segments[j].y} #{j}')
                
                cvWidth, _ = EKF.covariance_matrix.shape
                
                # this is the predicted measurement based on where we think we are
                L_range = math.sqrt((EKF.system_state[2*i+3+0] - EKF.system_state[0])**2 + (EKF.system_state[2*i+3+1] - EKF.system_state[1])**2) 
                L_bearing = math.atan((EKF.system_state[2*i+3+1] - EKF.system_state[1])/(EKF.system_state[2*i+3+0]-EKF.system_state[0])) - EKF.system_state[2] 

                # predicted landmark measurement matrix
                h = np.array([L_range,L_bearing],'f')
                
                # create jacobian of the measurement model
                A = (EKF.system_state[0] - EKF.system_state[2*i+3+0])/L_range
                B = (EKF.system_state[1] - EKF.system_state[2*i+3+1])/L_range
                C = 0
                D = (EKF.system_state[1] - EKF.system_state[2*i+3+1])/L_range**2
                E = (EKF.system_state[0] - EKF.system_state[2*i+3+0])/L_range**2
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
                
                
                # this is the measurement of where the landmark has been detected
                L_range = math.sqrt((EKF.landmarks[i][0] - EKF.system_state[0])**2 + (EKF.landmarks[i][1] - EKF.system_state[1])**2) 
                L_bearing = math.atan((EKF.landmarks[i][1] - EKF.system_state[1])/(EKF.landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
                Z = np.array([L_range,L_bearing],'f')
                

                # calculate kalman gain for this landmark
                R = np.array([[L_range/100,0],[0,math.radians(1.0)*L_bearing]],'f') 
                K_gain = EKF.covariance_matrix @ J_measurement.transpose() @ np.linalg.inv(np.eye(2,2) @ R @ np.eye(2,2).transpose() + J_measurement @ EKF.covariance_matrix @ J_measurement.transpose())    
                print(K_gain)
                EKF.system_state += np.dot(K_gain , (Z-h))
                EKF.covariance_matrix = np.dot((np.eye(cvWidth,cvWidth) - np.dot(K_gain,J_measurement)) , EKF.covariance_matrix)
                # print(EKF.covariance_matrix)
                print("=====")

        '''
        step 3: Update covariance matrix and system state to include new landmarks
        '''
        
        J_xr = np.array(([1,0,-dy],[0,1,dx]),'f')
        J_z = np.array(([math.cos(EKF.system_state[2]),math.sin(EKF.system_state[2])],[-dt*math.sin(EKF.system_state[2]), dt*math.cos(EKF.system_state[2])]),'f')
        
        for i in range(int((len(EKF.system_state)-3)/2),len(EKF.landmarks)):
            EKF.covariance_matrix = np.pad(EKF.covariance_matrix,[(0,2),(0,2)],mode='constant')
            matW,matH = EKF.covariance_matrix.shape

            # add landmark to system state vector
            EKF.system_state = np.append(EKF.system_state,EKF.landmarks[i][0])
            EKF.system_state = np.append(EKF.system_state,EKF.landmarks[i][1])

            # calculate R matrix for current landmark
            L_range = math.sqrt((EKF.landmarks[i][0] - EKF.system_state[0])**2 + (EKF.landmarks[i][1] - EKF.system_state[1])**2) 
            L_bearing = math.atan((EKF.landmarks[i][1] - EKF.system_state[1])/(EKF.landmarks[i][0]-EKF.system_state[0])) - EKF.system_state[2] 
            R = multi_dot((np.identity(2),np.array([[L_range/100,0],[0,L_bearing*2*math.pi/360]],'f'),np.identity(2)))

            # extend covariance matrix and add the new landmarks covariance
            P_n1n1 = multi_dot((J_xr,EKF.covariance_matrix[0:3,0:3],J_xr.transpose())) + multi_dot((J_z,R,J_z.transpose()))
            EKF.covariance_matrix[matH-2:matH,matW-2:matW] = P_n1n1
            
            # add the robot - landmark covariance to the covariance matrix
            P_Rn1 = np.dot(EKF.covariance_matrix[0:3,0:3],J_xr.transpose())
            EKF.covariance_matrix[0:3,matW-2:matW] = P_Rn1

            # add landmark robot covariance to covariance matrix
            EKF.covariance_matrix[matH-2:matH,0:3] = P_Rn1.transpose()

            # add the landmark - landmark covariances to the remaining empty spots in the matrix
            for j in range(1,int((len(EKF.system_state)-3)/2)+1):  
                
                M = np.dot(J_xr,EKF.covariance_matrix[j*3:j*3+2,0:2].transpose())
                EKF.covariance_matrix[j*3:j*3+2,matH-3:matH] = M
                EKF.covariance_matrix[matW-3 :matW,j*3:j*3+2] = M.transpose()

        # set all the new landmarks as previous landmarks for the next time step
        EKF.prev_landmarks = np.array(copy.deepcopy(EKF.landmarks),'f')
        

        # send robots position to robot_pos topic
        msg = Pose()
        msg.orientation.x = float(EKF.system_state[0])
        msg.orientation.y = float(EKF.system_state[1])
        msg.orientation.z = float(EKF.system_state[2])
        
        publisher.publish(msg)
        
        EKF.xlocations.append(EKF.system_state[0])
        EKF.ylocations.append(EKF.system_state[1])

        # self.ax.plot(EKF.xlocations,EKF.ylocations,'g.')
        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
        EKF.prev_msg = msg