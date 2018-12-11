#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import rospy
from geometry_msgs.msg import Twist

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import math
import numpy as np
import sys, select, termios, tty

import time

from station_handler import DockingStation

# Turtlebot3 Maximum Speeds
MAX_LIN_VEL = 0.6
MAX_ANG_VEL = 0.4

# Maximum and Minimum Distance to be able to make dataset 
MIN_DISTANCE = 0.3       # 0.3 meter
MAX_DISTANCE = 1.5       # 1.5 meter


MAX_SPEED_RATIO = 0.3	# MAX_LIN_VEL * MAX_SPEED_RATIO

MAX_CAM_ANGLE = 0.3		# mamximum angle which the bot can see the station 
MAX_SAMPLING_IMAGE_NUM = 20 # maximum number of sampling images each (position, angle)

class AutoDockingDatasetGeneration(object):
	''' Generate dataset for autodocking training'''

	def __init__(self):

		self.labels = []
		self.images = []

		rospy.init_node('turtlebot3_autdocking_dataset_gathering')
		self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10) 
		
		self.station = DockingStation()
		self.station.get_position()
		
		self.cvBridge = CvBridge()
		self.sub_image_original = rospy.Subscriber('/camera/rgb/image_raw', Image, self.__callback_image_captured__, queue_size = 1)

		self.path = self._save_path()
		self.no_image_taken = False
		

	def __to_cv_image__(self, msg_img, img_show=True):
		''' 
			Convert ROS image to opencv image
		'''
		img = self.cvBridge.imgmsg_to_cv2(msg_img, "bgr8")
		img = cv2.resize(img, (100, 100))

		if img_show:
			cv2.imshow('turtlebot3 buger',img)
			cv2.waitKey(10)

		return img


	def __callback_image_captured__(self, msg_img):
		''' 
			Callback function to get an image 
		'''
		
		#angle = abs(self.station.compute_relative_angle())
		if self.no_image_taken \
			or self.station.is_existed == False \
			or self.station.min_front_dis <= MIN_DISTANCE \
			or self.station.center_laser_dis > MAX_DISTANCE:# \
			#or angle > 0.005:
				time.sleep(0.01)
				return	

		label = self.station.relative_robot_pos()
		if label[1] >= MIN_DISTANCE: 				# y-axis needs to be bigger than MIN_DISTANCE
			image = self.__to_cv_image__(msg_img)

			self.labels.append(label)
			self.images.append(image)

		time.sleep(0.05)	


	def _collision_situation(self):
		front_dis = self.station.min_front_dis
		left_dir = self.station.go_left
		if front_dis > MIN_DISTANCE:
			return False

		ang = -1. 
		if not left_dir: ang = ang * -1.0
		vel = -.1
		self.__send_robot_speed__(ang, vel, boost_ang=True)

		return True


	def _too_far_from_station(self):
		if self.station.center_laser_dis > MAX_DISTANCE + 0.2:
			self.__send_robot_speed__(0.0, 1.0, boost=False)
			return True
		return False


	def __send_robot_speed__(self, ang=0.0, lin=0.0, boost=False, boost_ang=False):
		vel_cmd = Twist()

		ang = np.clip(ang, -1.0, 1.0)
		vel_cmd.angular.z = ang*MAX_ANG_VEL
		if boost_ang: vel_cmd.angular.z *= 5.5

		lin = np.clip(lin, -1.0, 1.0)
		vel_cmd.linear.x = lin*MAX_LIN_VEL
		if boost: vel_cmd.linear.x *= 1.5

		self.pub_cmd_vel.publish(vel_cmd)

		return vel_cmd.angular.z, vel_cmd.linear.x


	def __straight_move__(self):
		front_dis = self.station.min_front_dis
		vel = np.random.choice([front_dis, 0.0])
		self.__send_robot_speed__(0.0, vel)


	def __random_move__(self):
		ang = np.random.choice([1.0,-1.0])
		ang = ang* np.random.rand()
		vel = np.clip(np.random.rand(), 0.01, MAX_SPEED_RATIO)
		self.__send_robot_speed__(ang, vel)


	def _save_path(self):
		path = os.path.dirname(os.path.realpath(__file__))
		path = path.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
							'turtlebot3_machine_learning/turtlebot3_auto_docking/dataset/')
		if not os.path.exists(path):
			os.makedirs(path)
			print("Successfully created save folder: ", path)
		return path


	def __build_training_dataset__(self):
		''' 
			Build training dataset 

			Each image is saved in the folder named with label values (position and heading)

			Position and heading value is rounded in -1 digit
		'''

		images = self.images
		labels = self.labels
		self.labels = []
		self.images = []
		self.data = []

		if len(images) == 0 or len(images) != len(labels): return 

		self.no_image_taken = True

		cnt = 0
		for image, label in zip(images,labels):

			label = np.round(label,1)
			if label[2] == 0: label[2] = 0.0	# to remove -0.0

			folder_path = self.path + str(label[1]) + "_" + str(label[0]) + "_" + str(label[2])
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)
				print("Successfully created save folder: ", folder_path)

			file_list = os.listdir(folder_path)

			# each dir has no more than 1MAX_SAMPLING_IMAGE_NUM images
			if len(file_list) >= MAX_SAMPLING_IMAGE_NUM:
				remove_index = np.random.randint(2, size=len(file_list))
				for i, image_name in enumerate(file_list):
					if remove_index[i] == 1:
						os.remove(folder_path+'/'+ image_name)
				#continue

			file_name = "/" + str(time.time()) + "."+ str(cnt) + ".jpg"

			cv2.imwrite(folder_path+file_name, image)
			print('saved', file_name)
			cnt += 1

		self.no_image_taken = False


	def __getKey__(self, settings):
		tty.setraw(sys.stdin.fileno())
		rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
		if rlist:
			key = sys.stdin.read(1)
		else:
			key = ''

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
		return key


	def __key_handler__(self, settings):
		try:
			key = self.__getKey__(settings)
			if key == '\x03': 
				print("key event")
				self.__send_robot_speed__(0.0,0.0)
				return False
		except:
			pass #print e
		finally:
			self.__send_robot_speed__(0.0,0.0)

		return True


	def run(self):
		''' Main Loop '''

		settings = termios.tcgetattr(sys.stdin)
		while True:

			if self._collision_situation() or self._too_far_from_station():
				self.no_image_taken = True
			else:
				if self.no_image_taken or abs(self.station.compute_relative_angle()) > MAX_CAM_ANGLE:
					self.__send_robot_speed__(0.0, 0.0)
					self.no_image_taken = True
					self.station.align_and_face_robot()
					self.no_image_taken = False

				self.__random_move__()
				self.__build_training_dataset__()

			if not self.__key_handler__(settings): break

		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__=="__main__":
	AutoDockingDatasetGeneration().run()

# end of file