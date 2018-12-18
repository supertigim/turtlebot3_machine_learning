#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
from keras import applications, regularizers
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.optimizers import SGD, Adam 
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Activation, Conv2D, MaxPool2D
from keras.utils.training_utils import multi_gpu_model
from keras.layers import Lambda, concatenate
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.utils.vis_utils import plot_model
from keras import backend as K

import tensorflow as tf
from dataset_preparation import TRAIN_DATA_PATH, VAL_DATA_PATH, N_TRAIN_SAMPLES, N_VAL_SAMPLES, CLASSES

############################################
# Configurations 
############################################

# path to the model weights files.
__PATH__                = os.path.dirname(os.path.realpath(__file__))
__PATH__                = __PATH__.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                            'turtlebot3_machine_learning/turtlebot3_auto_docking')

SAVE_FOLDER             = __PATH__ + '/pos_prediction_model'

#LOG_DIR                 = './log'

SAVE_PERIOD             = 10

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT   = 100, 100

EPOCHS                  = 200
BATCH_SIZE              = 64
VAL_BATCH_SIZE          = BATCH_SIZE//2

TRANSFERED_MODEL_UPDATE = True

LOAD_EPOCH              = 0 # must be less than epochs
N_GPU                   = 2 # Number of GPU

TOP_HIDDEN_L            = 512#1024

SAVE_MODEL_GRAPH        = False

WORK_THREAD             = 16

POS_LABEL_FILE          = '/autodock_pos_labels'


############################################
# Functions 
############################################

class ModelMGPU(Model):
    ''' 
            This class from Keras team enables a model to save properly in multiple GPUs training environment
            so that the model can be loaded on only CPU environment with no problem.    
    '''
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 95:
        lr = 1e-6
    elif epoch > 80:
        lr = 5e-6
    elif epoch > 65:
        lr = 1e-5
    elif epoch > 50:
        lr = 5e-5
    elif epoch > 35:
        lr = 1e-4
    elif epoch > 20:
        lr = 5e-4
    print('Learning rate: ', lr)
    return lr


def bulid_simple_net(width, height, RGB, ouput_classes, load_epoch = 0):

    MODEL_W_PATH = SAVE_FOLDER + '/simple_nn_weights_'

    channel = 3 if RGB == True else 1

    dev = "/cpu:0" if N_GPU != 1 else "/gpu:0"
    with tf.device(dev):

        if load_epoch != 0:
            model = load_model(MODEL_W_PATH + str(load_epoch) + '.h5')
        else:
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(width, height, channel), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

            model.add(MaxPool2D((2, 2)))
            model.add(Dropout(0.20))

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

            model.add(MaxPool2D((2, 2)))
            model.add(Dropout(0.30))

            model.add(Conv2D(128, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))

            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.40))

            model.add(Flatten())
            model.add(Dense(TOP_HIDDEN_L))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.50))

            model.add(Dense(ouput_classes
                                , kernel_regularizer=regularizers.l2(1e-4)
                                , activation='softmax'
                                , name = 'predictions'))

    if N_GPU > 1: model = ModelMGPU(model, N_GPU)

    # compile the model with a Adam optimizer and dynamic learning rate.
    model.compile(loss='categorical_crossentropy', \
                  optimizer=Adam(lr=lr_schedule(load_epoch)), \
                  #optimizer=SGD(lr=1e-4, momentum=0.9), \
                  metrics=['accuracy'])

    model.summary()

    return model, MODEL_W_PATH



def bulid_keras_pretrained_net(net_name, width, height, RGB, ouput_classes, load_epoch = 0):
    ''' 
            Build NASNET which is able to save weights in one file
    '''

    MODEL_W_PATH = SAVE_FOLDER + '/mobilenetv2_weights_'

    channel = 3 if RGB == True else 1

    dev = "/cpu:0" if N_GPU != 1 else "/gpu:0"
    with tf.device(dev):

        if load_epoch != 0:
            model = load_model(MODEL_W_PATH + str(load_epoch) + '.h5')
        else:
            # this could also be the output a different Keras model or layer
            input_tensor = Input(shape=(width, height, channel))  # this assumes K.image_data_format() == 'channels_last'

            if net_name == 'Xception':
                MODEL_W_PATH = SAVE_FOLDER + '/xception_weights_'
                base_model = applications.xception.Xception(input_tensor=input_tensor,weights='imagenet', include_top=False)
            elif net_name == 'MobileNet':
                MODEL_W_PATH = SAVE_FOLDER + '/mobilenetv2_weights_'
                base_model = applications.mobilenet_v2.MobileNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
            elif net_name == 'NasnetMoblie':
                MODEL_W_PATH = SAVE_FOLDER + '/nasnetmobile_weights_'
                base_model = applications.NASNetMobile(input_tensor=input_tensor,weights='imagenet', include_top=False)
            else:
                raise ValueError(net_name, ' is not supported!')

            x = base_model.output

            ## add a global spatial average pooling layer
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            #x = Flatten()(x)

            x = Dense(TOP_HIDDEN_L
                    ,name='hidden_l')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)

            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(ouput_classes
                                , kernel_regularizer=regularizers.l2(1e-4)
                                , activation='softmax'
                                , name = 'predictions')(x)

            # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)

            # set if some weights are trainable or not
            for layer in model.layers[:len(base_model.layers)]:
                layer.trainable = TRANSFERED_MODEL_UPDATE

    #model = multi_gpu_model(model, gpus=N_GPU)
    if N_GPU > 1: model = ModelMGPU(model, N_GPU)

    # compile the model with a Adam optimizer and dynamic learning rate.
    model.compile(loss='categorical_crossentropy', \
                  optimizer=Adam(lr=lr_schedule(load_epoch)), \
                  #optimizer=SGD(lr=1e-4, momentum=0.9), \
                  metrics=['accuracy'])

    model.summary()

    return model, MODEL_W_PATH



class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x) - 0.5) * 2.
        return x


def retrieve_pos_and_dir(folder_name):
    '''
        retrieve robot position(x,y) and relative direction to the docking station 
    '''

    data = folder_name.split('_')
    return [float(data[1]), float(data[0]), float(data[2])] # x , y, relative dir


def store_labels(gen, save_file=None):
    '''
        Crate numpy label dictionaly and store it 
    '''

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print("Successfully created " + SAVE_FOLDER + ' directory')

    if save_file == None:
        save_file = SAVE_FOLDER + POS_LABEL_FILE

    labels = (gen.class_indices)
    label_np_dic = np.ndarray(shape=(len(labels), 3),dtype=float)

    for folder_name,index in labels.items():
        data = retrieve_pos_and_dir(folder_name)
        label_np_dic[index][0] = data[1]
        label_np_dic[index][1] = data[0]
        label_np_dic[index][2] = data[2]

    np.save(save_file, label_np_dic)

    print('saved labels as numpy array ', save_file)


def load_labels(load_file=None):
    ''' 
        Load numpy label dictionary
    '''

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print("Successfully created " + SAVE_FOLDER + ' directory')

    if load_file == None:
        load_file = SAVE_FOLDER + POS_LABEL_FILE + '.npy'

    return np.load(load_file)
    

def load_dataset(width, height, train_sample_path, validataion_sample_path, batch_size, val_batch_size):
    '''
        prepare data augmentation configuration
    '''
    data_gen_args = dict(rescale=1. / 255,
                         #featurewise_center=True,
                         #featurewise_std_normalization=True,
                         zca_whitening=True,
                         #fill_mode="nearest"
                         )

    train_datagen = ImageDataGenerator(**data_gen_args)

    train_generator = train_datagen.flow_from_directory(
        train_sample_path,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=np.random.seed())

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validataion_sample_path,
        target_size=(height, width),
        batch_size=val_batch_size,
        class_mode='categorical')

    store_labels(train_generator)

    return train_generator, validation_generator


def train_model():
    '''
       Model Training
    '''

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        print("Successfully created " + SAVE_FOLDER + ' directory')

    #if not os.path.exists(LOG_DIR):
    #    os.makedirs(LOG_DIR)
    #    print("Successfully created " + LOG_DIR + ' directory')


    #model, model_path = bulid_simple_net(IMG_WIDTH, IMG_HEIGHT, True, CLASSES, LOAD_EPOCH)
    model, model_path = bulid_keras_pretrained_net('MobileNet', IMG_WIDTH, IMG_HEIGHT, True, CLASSES, LOAD_EPOCH)

    train_dataset, val_dataset = load_dataset(IMG_WIDTH
                                            , IMG_HEIGHT
                                            , TRAIN_DATA_PATH
                                            , VAL_DATA_PATH
                                            , BATCH_SIZE
                                            , VAL_BATCH_SIZE)

    
    if SAVE_MODEL_GRAPH:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
    callbacks = [LearningRateScheduler(lr_schedule),
                ModelCheckpoint(model_path+'{epoch:d}_{val_loss:.2f}.h5'
                                , monitor='val_loss'
                                , verbose=0
                                , save_best_only=False
                                , save_weights_only=False
                                , mode='auto'
                                , period=SAVE_PERIOD)]

    # fine-tune the model
    model.fit_generator(
        generator=train_dataset,
        steps_per_epoch=N_TRAIN_SAMPLES//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_dataset,
        validation_steps=N_VAL_SAMPLES//VAL_BATCH_SIZE,
        initial_epoch = LOAD_EPOCH,
        callbacks=callbacks, 
        verbose=1,
        workers=WORK_THREAD)





if __name__ == '__main__':
    train_model()

# end of file