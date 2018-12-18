#!/usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from math import pi
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply

FRONT_HALF_DEGREE = 35          # front angle ranged from -FRONT_HALF_DEGREE to FRONT_HALF_DEGREE
STATION_POS = 0.84              # set by wall position in small square model in turtlebot3/simulations/gazebo

STATION_NAME = 'station'        # displayed on gazebo 

class DockingStation():
    ''' Docking Station Handling Class '''

    def __init__(self):
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.Path.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                           'turtlebot3_machine_learning/turtlebot3_auto_docking/models/docking_mark/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        self.is_up_or_bottom = True
        
        self.station_position = Pose()
        self.station_name = STATION_NAME

        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.__check_if_existed__, queue_size=1)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.__get_robot_odometry__, queue_size=1)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.__get_scan__, queue_size=1)
        self.is_existed = False
        self.index = 0

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.center_laser_dis = 0.0
        self.min_front_dis = float("inf")
        self.go_left = True

        self.r = rospy.Rate(10)

        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)


    def __get_robot_odometry__(self, odom):
        ''' save robot position and yaw '''
        self.robot_x = odom.pose.pose.position.x
        self.robot_y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        self.robot_yaw = math.atan2(2*(q.z*q.w + q.x*q.y) , 1 - 2*(q.y**2 + q.z**2)) 


    def __get_scan__(self, scan):
        ''' 
            handle scan data (currently number is 241) 

            - update cneter laser distance
            - update minimum distance in front 
            - update which side has a closer obstacle
        '''
        scan_length = len(scan.ranges)
        self.center_laser_dis = scan.ranges[scan_length//2]
        self.min_front_dis = min(scan.ranges[scan_length//2-FRONT_HALF_DEGREE:scan_length//2+FRONT_HALF_DEGREE]) 

        left = min(scan.ranges[:scan_length//2-FRONT_HALF_DEGREE])
        right = min(scan.ranges[scan_length//2+FRONT_HALF_DEGREE:])

        self.go_left = True if left > right else False


    def __check_if_existed__(self, model):
        ''' is station existed? '''
        self.is_existed = False
        for i in range(len(model.name)):
            if model.name[i] == self.station_name:
                self.is_existed = True
                return

    def ros_time(self):
        return rospy.Time.now().to_sec()


    def ros_sleep(self):
        self.r.sleep()
        

    def compute_relative_angle(self):
        ''' 
            return is 0.0 if the bot faces the station exactly 
        '''
        goal_angle = math.atan2(self.station_position.position.y - self.robot_y
                                ,self.station_position.position.x - self.robot_x)
        heading = self.robot_yaw - goal_angle
        if heading > pi:   heading -= 2 * pi
        elif heading < -pi:    heading += 2 * pi

        #print("Heading:", heading)
        return heading


    def relative_robot_pos(self):
        '''
            station pos is assumed as [0.0, 0.0]
            yaw(face) of station is y-axis 
        '''

        if self.station_position.position.y == -STATION_POS:
            x = self.robot_x - self.station_position.position.x
            y = self.robot_y - self.station_position.position.y
        elif self.station_position.position.y == STATION_POS:
            x = self.station_position.position.x - self.robot_x
            y = self.station_position.position.y - self.robot_y
        else:

            if self.station_position.position.x == STATION_POS:
                x = self.station_position.position.x - self.robot_x
                y = self.robot_y-self.station_position.position.y
            else:
                x = self.robot_x - self.station_position.position.x
                y = self.station_position.position.y - self.robot_y
            x , y = y , x

        #return [x/MAX_DIS_TO_CONSIDER,y/MAX_DIS_TO_CONSIDER,self.compute_relative_angle()/pi]
        return [x,y,self.compute_relative_angle()]


    def create_station(self):
        ''' create a station '''
        for _ in range(5):
            if not self.is_existed:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.station_name, self.model, 'robotos_name_space', self.station_position, "world")
                self.align_and_face_robot()
                break
            
            time.sleep(0.05)


    def delete_station(self):
        ''' Delete existed station '''
        for _ in range(5):
            if self.is_existed:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                self.is_existed = False
                del_model_prox(self.station_name)
                #print('deleted')
                break
            
            time.sleep(0.05)


    def __station_orientation__(self):
        ''' 
            return appropriate station orientation 
            
            Make sure that this funcation should come after __new_position__(), because this depends on variable 'is_up_or_bottom'
        '''

        pose = Pose()

        #print(self.is_up_or_bottom)

        if not self.is_up_or_bottom:
            q = quaternion_from_euler(0.0,0.0,pi/2) 

            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

        return pose.orientation

    def place_randomly(self):
        ''' 
            Move station aligning and facing robot exactly 
        '''
        MAX_DYNAMIC_POS = 0.7

        self.is_up_or_bottom = True if random.random() > 0.5 else False 
        bottom_or_right = 1.0 if random.random() > 0.5 else -1.0
        top_or_left = 1.0 if random.random() > 0.5 else -1.0

        x = STATION_POS * bottom_or_right
        y = (random.random() % MAX_DYNAMIC_POS) * top_or_left
        if not self.is_up_or_bottom: x,y = y,x

        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(model.name)):
            if model.name[i] == self.station_name:
                mark = ModelState()
                mark.model_name = model.name[i]
                mark.pose = model.pose[i]

                mark.pose.position.x, mark.pose.position.y = x, y

                self.station_position.position.x = mark.pose.position.x
                self.station_position.position.y = mark.pose.position.y
                
                mark.pose.orientation = self.__station_orientation__()
                self.pub_model.publish(mark)

                time.sleep(0.02)
                break


    def align_and_face_robot(self):
        ''' 
            Move station aligning and facing robot exactly 
        '''
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(model.name)):
            if model.name[i] == self.station_name:
                mark = ModelState()
                mark.model_name = model.name[i]
                mark.pose = model.pose[i]

                mark.pose.position.x, mark.pose.position.y = self.__new_position__()

                self.station_position.position.x = mark.pose.position.x
                self.station_position.position.y = mark.pose.position.y
                
                mark.pose.orientation = self.__station_orientation__()
                self.pub_model.publish(mark)

                time.sleep(0.02)
                break
                    

    def __compute_center_ob_pos__(self):
        ''' 
            return front obstacle position by robot yaw and center laser distance 
        '''

        pos = [self.center_laser_dis,0]

        sin_theta = math.sin( self.robot_yaw )
        cos_theta = math.cos( self.robot_yaw )

        a0 = pos[0]*cos_theta - pos[1]*sin_theta
        a1 = pos[0]*sin_theta + pos[1]*cos_theta

        a = [a0+self.robot_x, a1+self.robot_y]
        return a


    def __new_position__(self):
        ''' 
            calculate a new possible position using laser scan on the bot 
        '''

        FIXED_POS = 0.87
        MAX_DYNAMIC_POS = 0.7

        ob = self.__compute_center_ob_pos__()
        #print(ob)

        if abs(abs(ob[0]) - FIXED_POS) < abs(FIXED_POS-MAX_DYNAMIC_POS):
            if ob[0] > 0:               # upper side wall
                x = STATION_POS
            else:                       # buttom side wall
                x = -STATION_POS
            y = min(MAX_DYNAMIC_POS,max(-MAX_DYNAMIC_POS, ob[1]))
            self.is_up_or_bottom = True
        else:
            if ob[1] > 0:               # left side wall
                y = STATION_POS
            else:                       # right side wall
                y = -STATION_POS
            x = min(MAX_DYNAMIC_POS,max(-MAX_DYNAMIC_POS, ob[0]))
            self.is_up_or_bottom = False

        #print('new station pos: ',x,y)
        return x, y
            

    def get_position(self):#, position_check=False):
        ''' 
            Delete a old station, create a new one, and return (x,y) of the station
        '''

        self.delete_station()

        x, y = self.__new_position__()
        self.station_position.position.x = x
        self.station_position.position.y = y

        self.create_station()

        return self.station_position.position.x, self.station_position.position.y


def main():
    rospy.init_node('turtlebot3_handle_station')
    try:
        station = DockingStation()
        station.get_position()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

# end of file