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

# path to the model weights files.
__PATH__        = os.path.dirname(os.path.realpath(__file__))
__PATH__        = __PATH__.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                            'turtlebot3_machine_learning/turtlebot3_auto_docking')
VAL_DATA_PATH  =  __PATH__ + '/data/validation'

NUM             = 160
MODEL_W_PATH    = SAVE_FOLDER + '/simple_nn_weights_' + str(NUM) + '.h5'

DETECTION_THRESHOLD_ACCURACY = 0.05     # above 10%


class ModelEvaluation(AutoDockingDataGathering):
    def __init__(self):
        super(ModelEvaluation, self).__init__()

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
        label[0], label[1] = label[1], label[0]
        if prob > DETECTION_THRESHOLD_ACCURACY:
            pred = self.label_dic[ind]
            pred = [str(a) for a in pred]
            label = [str(a) for a in label]
            print('Predicted('+str(round(prob,2)*100)+'%):['+','.join(pred)+'] | Label:['+','.join(label)+']')


    def run(self):
        ''' Main Loop '''
        settings = termios.tcgetattr(sys.stdin)
        
        while True:
            key = self.__getKey__(settings)

            if key == '\x03': 
                print("Bye~")
                return False
            elif key == 'n':
                self.no_image_taken = True
                self.station.align_and_face_robot()
                self.no_image_taken = False

            self.__predict_pos__()

            time.sleep(0.005)
                
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