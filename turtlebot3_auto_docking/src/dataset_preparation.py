# -*- coding: utf-8 -*-
import os
import random
import shutil
import numpy as np

############################################
# Configurations 
############################################

__PATH__                = os.path.dirname(os.path.realpath(__file__))
__PATH__                = __PATH__.replace('turtlebot3_machine_learning/turtlebot3_auto_docking/src',
                                            'turtlebot3_machine_learning/turtlebot3_auto_docking')

TRAIN_DATA_PATH         = __PATH__ + '/data/train'
VAL_DATA_PATH           = __PATH__ + '/data/validation'


SOURCE_PATH             = __PATH__ + '/dataset'


SPLIT_RATIO             = 0.1           # 0.9 : Training | 0.1 : Testing 



# These valuse come after excuting the below function 
N_TRAIN_SAMPLES         = 54213
N_VAL_SAMPLES           = 6332

CLASSES                 = 1464

############################################
# Functions 
############################################


def split_dataset_into_train_and_validation_samples(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    '''
        Split Dataset into test and 
    '''
    # Recreate testing and training directories
    if not os.path.exists(testing_data_dir):
        os.makedirs(testing_data_dir)
        print("Successfully created validation folder")
    elif testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if not os.path.exists(training_data_dir):
        os.makedirs(training_data_dir)
        print("Successfully created training folder")
    elif training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    num_classes = 0
    for subdir, _, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        if category_name == os.path.basename(all_data_dir):
            continue
        print(category_name + " in " + os.path.basename(all_data_dir))
        num_classes += 1

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

        if not len(os.listdir(testing_data_category_dir)):
            input_file = os.path.join(subdir, files[-1])
            shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + files[-1])
            num_testing_files += 1


    print("\n\n#######################################################")
    print("Number of dataset categories is " + str(num_classes))
    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


def main():
    '''
        Main Function 
    '''
    SPLIT_RATIO = 0.1
    split_dataset_into_train_and_validation_samples(SOURCE_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, SPLIT_RATIO)




if __name__ == '__main__':
    main()


# end of file