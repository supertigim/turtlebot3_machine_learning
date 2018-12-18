# -*- coding:utf-8 -*-
#!/usr/bin/env python

import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from data_gathering import AutoDockingDataGathering, MAX_LIN_VEL, MAX_ANG_VEL
from pos_predictor_training import SAVE_FOLDER, IMG_WIDTH, IMG_HEIGHT, load_labels

import numpy as np
import sys, select, termios, tty
import math 

# path to the model weights files.
__PATH__        = os.path.dirname(os.path.realpath(__file__))
__PATH__        = __PATH__.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                            'turtlebot3_machine_learning/turtlebot3_auto_docking')
VAL_DATA_PATH  =  __PATH__ + '/data/validation'

NUM             = 110
MODEL_W_PATH    = SAVE_FOLDER + '/mobilenetv2_weights_' + str(NUM) + '.h5'

DETECTION_THRESHOLD_ACCURACY = 0.95    # above 95%


class ModelEvaluation(AutoDockingDataGathering):
    def __init__(self):
        super(ModelEvaluation, self).__init__()
        self.station.place_randomly()

        self.model = load_model(MODEL_W_PATH)
        self.label_dic = load_labels()

        
    def __callback_image_captured__(self, msg_img):
        ''' 
            Callback function to get an image 
        '''

        if self.no_image_taken:
            time.sleep(0.01)
            return	

        label = self.station.relative_robot_pos()
        image = self.__to_cv_image__(msg_img)

        self.labels.append(label)
        self.images.append(image)

        time.sleep(0.005)


    def __go_ready_to_dock_position__(self, robot_pos):
        print('__face_ready_to_dock_position__ started')
        self.__send_robot_speed__(0.0, 0.0)
        self.station.ros_sleep()
        clockwise = True if robot_pos[0] > 0 else False
        
        ready_point = 0.90
        a = round(math.sqrt(robot_pos[0]**2 + robot_pos[1]**2),2)
        b = round(math.sqrt(robot_pos[0]**2 + (robot_pos[1]-ready_point)**2),2)
        #b = abs(robot_pos[0])
        c = ready_point # (0.0, 0.0) ~ (0.0, ready_point)

        if a+b <= c:
            if robot_pos[1] >= ready_point:
                ang = 0.0
            else:
                ang = math.pi
        else:
            ang = math.acos((a**2+b**2-c**2)/(2*a*b))
            print('a:',a, ' b:', b, ' c:',c, ' angle:', ang)

        robot_ang = robot_pos[2]
        if clockwise: robot_ang *= -1.0
        relative_angle = abs(ang + robot_ang)

        print('angle:', ang, ' robot angle:', robot_pos[2], ' relative_angle:', relative_angle) 

        # Checking if our movement is CW or CCW
        angular_speed = 1.0
        if clockwise: angular_speed = -abs(angular_speed)

        # 01.Rotation 
        print('01.Rotation ')
        t0 = self.station.ros_time()
        desired_angle = 0
        while(desired_angle < relative_angle):
            z, _ = self.__send_robot_speed__(ang=angular_speed, lin=0.0)
            self.station.ros_sleep()
            #print('desired_angle:', desired_angle, ' angular speed:', z)
            t1 = self.station.ros_time()
            desired_angle = abs(z*(t1-t0))
        self.station.ros_sleep()
        self.__send_robot_speed__(0.0, 0.0)
        
        # 02.Move to (0.0, 1.0)
        print('02.Move to <
        #BUGER_WHEEL_RADIU<0
        t0 = self.station.<
        distance_moved = 0<
        linear_speed = 0.25 # Ratio
        while(distance_moved < b):
            _ , lin_speed = self.__send_robot_speed__(ang=0.0, lin=linear_speed)
            self.station.ros_sleep()
            t1 = self.station.ros_time()
            distance_moved = abs(lin_speed*(t1-t0))
        self.__send_robot_speed__(0.0, 0.0)
        self.station.ros_sleep()

        # 03.Face the docking station
        print('03.Face the docking station') 

        if a+b <= c:
            if robot_pos[1] >= ready_point:
                ang = 0.0
            else:
                ang = math.pi
        else:
            relative_angle = math.acos((b**2+c**2-a**2)/(2*b*c))
        relative_angle = math.pi - relative_angle
        angular_speed *= -1.0

        t0 = self.station.ros_time()
        desired_angle = 0
        while(desired_angle < relative_angle):
            z, _ = self.__send_robot_speed__(ang=angular_speed, lin=0.0)
            self.station.ros_sleep()
            t1 = self.station.ros_time()
            desired_angle = abs(z*(t1-t0))
        self.station.ros_sleep()
        self.__send_robot_speed__(0.0, 0.0)

        self.no_image_taken = True
        self.station.place_randomly()
        self.no_image_taken = False

        self.labels = []
        self.images = []
        self.data = []

        print('__face_ready_to_dock_position__ ended')


    def __predict_pos__(self):
        images = self.images
        labels = self.labels

        self.labels = []
        self.images = []
        self.data = []
        
        if len(images) == 0 or len(images) != len(labels): return

        image = images[-1]
        label = labels[-1]

        data = np.array(image, dtype=np.float32)
        data.shape = (1,) + data.shape
        data /= 255.

        predictions = self.model.predict(data)[0]
        prob = np.max(predictions, axis=-1)
        ind = np.argmax(predictions, axis=-1)

        label = np.round(label,2)
        if prob > DETECTION_THRESHOLD_ACCURACY:
            pred = self.label_dic[ind]
            pred[0], pred[1] = pred[1], pred[0]

            pred_string = [str(a) for a in pred]
            label = [str(a) for a in label]
            print('\nPredicted('+str(round(prob,2)*100)+'%):['+','.join(pred_string)+'] | Label:['+','.join(label)+']')

            self.__go_ready_to_dock_position__(pred)

        self.no_image_taken = False


    def run(self):
        ''' Main Loop '''
        settings = termios.tcgetattr(sys.stdin)
        
        while True:
            key = self.__getKey__(settings)
            
            if not self._collision_situation():
                if key == '\x03': 
                    print("Bye~")
                    return False
                elif key == 'n':
                    self.no_image_taken = True
                    self.station.align_and_face_robot()
                    self.no_image_taken = False
                elif key == 'r':
                    self.no_image_taken = True
                    self.station.place_randomly()
                    self.no_image_taken = False

                self.__predict_pos__()
                self.__random_move__(boost_ang=True)

            time.sleep(0.005)
        
        self.__send_robot_speed__()
                
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main():
    ModelEvaluation().run()


def evaluate_model_with_validation_dataset():

    model = load_model(MODEL_W_PATH)

    # Generate dataset from a path
    BATCH_SIZE = 100
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        VAL_DATA_PATH,
        batch_size = BATCH_SIZE,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical'
    )

    # Create Label map
    label_map = (test_generator.class_indices)
    label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
    total_accuray  = 0
    total_num = 0
	
    while True:
        # Take first one of dataset
        data, labels = test_generator.next()
        #if data == None: break
        labels = np.argmax(labels, axis=-1) 
        labels = [label_map[k] for k in labels]

        # Predict
        predictions = model.predict(data)
        total_num += len(predictions)
        if len(predictions) != BATCH_SIZE : break

        # Label predictions
        predictions = np.argmax(predictions, axis=-1) #multiple categories
        predictions = [label_map[k] for k in predictions]

        # Visualize results
        cnt = 0
        for p, l in zip(predictions, labels):
            result = True if p == l else False
            if result: 
                cnt += 1
                total_accuray += 1
            #print(p, " : ", l, " ", result)
        print("Accuracy:",str(cnt) + "/" + str(BATCH_SIZE), " ("+ str(100.0*cnt/BATCH_SIZE)+")")

    print("Overall Accuracy:",str(total_accuray) + "/" + str(total_num), " ("+ str(100.0*total_accuray/total_num)+")")

    
if __name__=="__main__":
    #evaluate_model_with_validation_dataset()
    main()

# end of file