# -*- coding:utf-8 -*-
#!/usr/bin/env python
import os 

import rospy
from geometry_msgs.msg import Twist, Point, Pose
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry

import cv2
from cv_bridge import CvBridge 

from gym import spaces 
import gym

import numpy as np 
import time
import math
import random

from data_gathering import MAX_LIN_VEL, MAX_ANG_VEL

ACT_DIM = 2                # Angular / Linear Velocities
OBS_SHAPE = (100,100,3)    # (100,100,3,)

COLLISON_DISTANCE = 0.2 
CAMERA_ON = True

STATION_NAME = 'station'        # displayed on gazebo 
ROBOT_NAME = 'turtlebot3_burger'

STATION_POS = 0.84              # set by wall position in small square model in turtlebot3/simulations/gazebo

class DockingStationGym(gym.Env):
    def __init__(self):
        # For OpenAI gym 
        self.__version__ = "0.1.0"
        high = np.ones(ACT_DIM)
        self.action_space = spaces.Box(-high, high,dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=OBS_SHAPE,dtype=np.float32)
        self.done = True
        self.captured_image = []  

        # For ROS 
        self.Path = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.Path.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                           'turtlebot3_machine_learning/turtlebot3_auto_docking/models/docking_mark/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        
        self.r = rospy.Rate(10)

        self.bot_name = ROBOT_NAME
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.station_name = STATION_NAME
        self.station_position = Pose()
        self.is_station_existed = False

        self.center_laser_dis = float('inf')

        self.cvBridge = CvBridge()
        self.sub_image_ = rospy.Subscriber('/camera/rgb/image_raw', Image, self.__callback_image_captured__, queue_size = 1)
        
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.__get_robot_odometry__, queue_size=1)
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.__check_if_station_existed__, queue_size=1)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.__get_scan__, queue_size=1)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)


        while self.done:
            time.sleep(0.01)
        time.sleep(1)
            

    def step(self, action, reset=False):
        self.__send_robot_speed__(*action)        
        self.r.sleep()

        reward, done = self.__get_reward__()
        return self.__get_current_state__(), reward, done, {}


    def reset(self):
        self.__send_robot_speed__()
        self.r.sleep()

        # locate the robot in the center on simulation
        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(model.name)):
            if model.name[i] == self.bot_name:
                bot = ModelState()
                bot.model_name = model.name[i]
                bot.pose = Pose()
                bot.pose.position.x = -0.2

                self.pub_model.publish(bot)
                self.r.sleep()
                break

        # randomly place the docking station in front of the bot
        self.create_station()
        self.place_station_in_front_randomly()

        return self.__get_current_state__()


    def render(self, mode='Docking_Station_Simulation', close=False):
        return


    def close(self):
        self.reset() 


    def create_station(self):
        ''' create a station '''
        for _ in range(2):
            if not self.is_station_existed:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.station_name, self.model, 'robotos_name_space', self.station_position, "world")
                break
            time.sleep(0.01)


    def place_station_in_front_randomly(self):
        ''' 
            Move station aligning and facing robot exactly 
        '''
        MAX_DYNAMIC_POS = 0.7

        x = STATION_POS
        top_or_left = 0.0#1.0 if random.random() > 0.5 else -1.0
        y = (random.random() % MAX_DYNAMIC_POS) * top_or_left

        model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        for i in range(len(model.name)):
            if model.name[i] == self.station_name:
                mark = ModelState()
                mark.model_name = model.name[i]
                mark.pose = model.pose[i]

                mark.pose.position.x, mark.pose.position.y = x, y

                self.station_position.position.x = mark.pose.position.x
                self.station_position.position.y = mark.pose.position.y
                
                #mark.pose.orientation = self.__station_orientation__()
                self.pub_model.publish(mark)

                time.sleep(0.05)
                break
                
    
    def __compute_relative_angle__(self):
        ''' 
        return is 0.0 if the bot faces the station exactly 
        '''
        goal_angle = math.atan2(self.station_position.position.y - self.robot_y
                                ,self.station_position.position.x - self.robot_x)
        heading = self.robot_yaw - goal_angle
        
        if heading > math.pi:   heading -= 2 * math.pi
        elif heading < -math.pi:    heading += 2 * math.pi

        #print("Heading:", heading)
        return heading


    def __get_reward__(self):
        MAX_REWARD = 5.0

        heading = abs(self.__compute_relative_angle__())
        is_exact_angle  = heading <= 0.008

        if self.center_laser_dis <= COLLISON_DISTANCE + 0.05 and is_exact_angle:
            print('################ Reach the docking station ##################')
            return MAX_REWARD, True

        if self.done:
            print('>> Crashed')
            return -MAX_REWARD, True

        reward_orientation = 1.0 if is_exact_angle else 0.0
        print('reward_orientation', reward_orientation, heading, self.center_laser_dis)
        reward_distance = max(0.0, min(1.0, 1.2 - self.center_laser_dis))
        reward = reward_orientation*reward_distance
        return reward, False


    def __check_if_station_existed__(self, model):
        ''' is station existed? '''
        self.is_station_existed = False
        for i in range(len(model.name)):
            if model.name[i] == self.station_name:
                self.is_station_existed = True
                return


    def __get_current_state__(self):
        while not len(self.captured_image):
            time.sleep(0.01)
        state = self.captured_image[-1]
        self.captured_image = []
        return state 


    def __get_robot_odometry__(self, odom):
        ''' save robot position and yaw '''
        self.robot_x = odom.pose.pose.position.x
        self.robot_y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation

        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        self.robot_yaw = math.atan2(2*(q.z*q.w + q.x*q.y) , 1 - 2*(q.y**2 + q.z**2)) 

        time.sleep(0.001)

    
    def __get_scan__(self, scan):
        ''' 
            handle scan data (currently number is 241) 

            - update cneter laser distance
            - update minimum distance in front 
            - update which side has a closer obstacle
        '''
        self.done = True if min(scan.ranges) < COLLISON_DISTANCE else False

        scan_length = len(scan.ranges)
        self.center_laser_dis = scan.ranges[scan_length//2]

        time.sleep(0.001)


    def __callback_image_captured__(self, msg_img):
        ''' 
            Convert ROS image to opencv image
        '''

        if self.done: time.sleep(0.001);return 

        img = self.cvBridge.imgmsg_to_cv2(msg_img, "bgr8")

        if CAMERA_ON:
            cv2.imshow('turtlebot3 buger',img)
            cv2.waitKey(10)

        # resize net input size 
        img = cv2.resize(img, (OBS_SHAPE[0], OBS_SHAPE[1]))
        img = img[...,::-1]  #RGB 2 BGR

        data = np.array(img, dtype=np.float32)
        data = data.transpose((0, 1, 2))
        data.shape = (1,) + data.shape
        data /= 255.

        self.captured_image.append(data)
        time.sleep(0.001)


    def __send_robot_speed__(self, ang=0.0, lin=-1.0):
        
        #print(ang, lin)
        ang = np.clip(ang, -1.0, 1.0)
        lin = np.clip(lin, -1.0, 1.0)
        lin = (1.0 + lin)/2.0           # changed with the range from 0.0 to 1.0

        vel_cmd = Twist()
        vel_cmd.angular.z = ang*MAX_ANG_VEL
        vel_cmd.linear.x = lin*MAX_LIN_VEL

        self.pub_cmd_vel.publish(vel_cmd)
        return vel_cmd.angular.z, vel_cmd.linear.x


if __name__ == '__main__':
    rospy.init_node('turtlebot3_autdocking_gym_test')
    print('turtlebot3 autodocking gym started...')

    EPISODE = 2
    MAX_STEPS = 50

    env = DockingStationGym()
    for e in range(EPISODE):
        s = env.reset()
        for step in range(MAX_STEPS):
            #print('step:{}'.format(step))
            action = env.action_space.sample()
            action[1] = (action[1]+1.0)/2.0

            action = [0.0, 0.25]                 # Go straight
            _,_,done,_ = env.step(action)
            if done: break

    env.close()
    print('turtlebot3 autodocking gym end...')


# end of file