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

import pickle

import tensorflow as tf
from dataset_preparation import TRAIN_DATA_PATH, VAL_DATA_PATH, N_TRAIN_SAMPLES, N_VAL_SAMPLES, CLASSES

############################################
# Configurations 
############################################

# path to the model weights files.
SAVE_FOLDER             = './save_model'
BASE_MODEL_W_PATH       = SAVE_FOLDER + '/nasnet_weights_'
TOP_MODEL_W_PATH        = SAVE_FOLDER + '/top_layer_'

FLOWER_RESNET50_W_PATH  = SAVE_FOLDER + '/flower_resnet50_'
VGG_FACE_MODEL_W_PATH   = SAVE_FOLDER + '/vggface_weights_'


XCEPTION_MODEL_W_PATH   = SAVE_FOLDER + '/xception_weights_'


#BASE_MODEL_W_PATH       = VGG_FACE_MODEL_W_PATH
BASE_MODEL_W_PATH       = XCEPTION_MODEL_W_PATH

SAVE_PERIOD             = 10

LOG_DIR                 = './log'

# dimensions of our images.
IMG_WIDTH, IMG_HEIGHT   = 100, 100

EPOCHS                  = 200
BATCH_SIZE              = 32
VAL_BATCH_SIZE          = BATCH_SIZE//2

TRANSFERED_MODEL_UPDATE = True

LOAD_EPOCH              = 0 # must be less than epochs
N_GPU                   = 2 # Number of GPU

TOP_HIDDEN_L            = 1024

SAVE_MODEL_GRAPH        = False

WORK_THREAD             = 16

POS_LABEL_PICKLE        = 'autodock_pos_labels.pickle'


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
    #if epoch > 8:
    #    lr = 1e-5
    if epoch > 120:
        lr = 5e-5
    elif epoch > 80:
        lr = 1e-4
    elif epoch > 40:
        lr = 5e-4
    print('Learning rate: ', lr)
    return lr


def bulid_simple_net(width, height, RGB, ouput_classes, load_epoch = 0):

    channel = 3 if RGB == True else 1

    dev = "/cpu:0" if N_GPU != 1 else "/gpu:0"
    with tf.device(dev):

        if load_epoch != 0:
            model = load_model(BASE_MODEL_W_PATH + str(load_epoch) + '.h5')
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
            model.add(Dense(512))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.50))

            model.add(Dense(ouput_classes
                                , kernel_regularizer=regularizers.l2(1e-3)
                                , activation='softmax'
                                , name = 'predictions'))

    if N_GPU > 1: model = ModelMGPU(model, N_GPU)

    # compile the model with a Adam optimizer and dynamic learning rate.
    model.compile(loss='categorical_crossentropy', \
                  optimizer=Adam(lr=lr_schedule(load_epoch)), \
                  #optimizer=SGD(lr=1e-4, momentum=0.9), \
                  metrics=['accuracy'])

    model.summary()

    return model



def bulid_keras_pretrained_net(net_name, width, height, RGB, ouput_classes, load_epoch = 0):
    ''' 
            Build NASNET which is able to save weights in one file
    '''
    channel = 3 if RGB == True else 1

    dev = "/cpu:0" if N_GPU != 1 else "/gpu:0"
    with tf.device(dev):

        if load_epoch != 0:
            model = load_model(BASE_MODEL_W_PATH + str(load_epoch) + '.h5')
        else:
            # this could also be the output a different Keras model or layer
            input_tensor = Input(shape=(width, height, channel))  # this assumes K.image_data_format() == 'channels_last'

            if net_name == 'Xception':
                print("Import Pre-trained Xception Network")
                base_model = applications.xception.Xception(input_tensor=input_tensor,weights='imagenet', include_top=False)
            else:
                base_model = applications.NASNetMobile(input_tensor=input_tensor,weights='imagenet', include_top=False)

            # add a global spatial average pooling layer
            x = base_model.output

            x = Flatten()(x)
            x = Dense(TOP_HIDDEN_L
                    #, kernel_regularizer=regularizers.l2(0.)
                    #, activity_regularizer=regularizers.l2(1e-4)
                    ,name='hidden_l')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)

            # and a logistic layer -- let's say we have 200 classes
            predictions = Dense(ouput_classes
                                , kernel_regularizer=regularizers.l2(1e-3)
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

    return model



class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x) - 0.5) * 2.
        return x


def store_labels(gen, save_file=POS_LABEL_PICKLE):
    '''
    # https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
    '''

    labels = (gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    #test()
    #test()ol=pickle.HIGHEST_PROTOCOL)
    #test()
    #test()
def load_labels(load_file=POS_LABEL_PICKLE):
    with open(load_file, 'rb') as handle:
        return pickle.load(handle)
    print('No '+load_file+' existed')
    return None
    

def load_dataset(width, height, train_sample_path, validataion_sample_path, batch_size, val_batch_size):
    '''
        prepare data augmentation configuration
    '''
    data_gen_args = dict(rescale=1. / 255,
                         #featurewise_center=True,
                         #featurewise_std_normalization=True,
                         #zca_whitening=True,
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

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print("Successfully created " + LOG_DIR + ' directory')


    model = bulid_simple_net(IMG_WIDTH, IMG_HEIGHT, True, CLASSES, LOAD_EPOCH)
    #model = bulid_keras_pretrained_net('Xception', IMG_WIDTH, IMG_HEIGHT, True, CLASSES, LOAD_EPOCH)

    train_dataset, val_dataset = load_dataset(IMG_WIDTH
                                            , IMG_HEIGHT
                                            , TRAIN_DATA_PATH
                                            , VAL_DATA_PATH
                                            , BATCH_SIZE
                                            , VAL_BATCH_SIZE)

    

    if SAVE_MODEL_GRAPH:
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        
    callbacks = [LearningRateScheduler(lr_schedule),
                ModelCheckpoint(BASE_MODEL_W_PATH+'{epoch:d}_{val_loss:.2f}.h5'
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


def test():
    '''
        Test Somthing
    '''
    pass



if __name__ == '__main__':
    #test()
    train_model()
# end of file