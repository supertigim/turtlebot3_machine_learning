## 1.Introduction  
Autonomous Docking Module is implemented using deep learning technology. 

Idea behind this task is that 3 three steps are required to do. 
1. Figure out where the turtlbebot3 is
    - Predict the position relative to the docking spot by using an image captured on top of the bot 
2. Situate the bot appropriate distance from the station and face it. 
    - Move the robot to (0, 1) pose, where pose of the docking station is (0, 0) 
3. Try to reach the spot without angle problem 
    - Use reinforcement learning to get to the spot (Input: an image, Output: angular/linear velocities)

First and second steps seem to be redundant when it comes to thinking about powerfullnes of RL at a glance. However, unlike expectation, only RL approach ends up failing to solve this taks, because of its patially obeserved environment problem.  


## 2.Prerequisites   

1.[ROS kinectic on ubunt 16.04](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
2.[Turtlebot3 Dependant ROS Packages](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
3.[Turtlebot3 Simulation ROS Package](http://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#simulation)  
4.OpenAI baselines for reinforcement learning  

**Anaconda environment**  
    $ conda create -n ros python=2
    $ source activate ros
    $(ros) pip install rosinstall msgpack empy defusedxml netifaces, tensorflow, keras, pillow  

**Gazebo Simulation** 

    $ source activate ros
    $(ros) export TURTLEBOT3_MODEL=burger       # only supported!!!
    $(ros) roslaunch turtlebot3_gazebo turtlebot3_docking_station.launch    # Gazebo simulation 

**Turtlebot3 controller**  

    $ source activate ros
    $(ros) export TURTLEBOT3_MODEL=burger  
    $(ros) roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch  


## 3.Dataset Preparation for pose prediction    

Posistion prediction model needs to be trained by dataset which include pairs of images and those position/yaw label.
Dataset are stored in 'dataset' folder where many folders named with position information are created and capture images belong to a dedicated folder automatically.  
  
First open two terminals,  

### > First terminal  
    # in case of using anaconda  
    $ source activate ros             # python2.7, pip install rosinstall msgpack empy defusedxml netifaces  

    # Create a gazebo environment
    $(ros) export TURTLEBOT3_MODEL=burger 
    $(ros) roslaunch turtlebot3_gazebo turtlebot3_docking_station.launch

### > Sencond terminal  
    # in case of using anaconda  
    $ source activate ros             # python2.7, pip install rosinstall msgpack empy defusedxml netifaces  

    # Capture an image while moving the bot randomly 
    cd /home/your_id/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_auto_docking/src  
    $(ros) python generate_dataset.py  

### > Environment Change 

Virtual environment in Gazebo is quite limited, so try to change it as many as possible while training. 
Here are two ways you can do.  

1. Put light on gazebo 
2. Change material of walls by editing turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_small_square/model.sdf  
    - Material list can be found in /usr/share/gazebo-7/media/materials/scripts/gazebo.material  
    - Find available one by keyword "material Gazebo/"  

## 4.Position Prediction Neural Network Training  

### > Building Dataset and Training ConvNet  

The gathered dataset need to be divided into two. One group called ‘training dataset’ is for training meanwhile other one called ‘testing dataset’ is to evaluate the model to see if training is done well. All dataset are placed in /data/train and /data/validation respectively. Dataset used here are in the [link](https://cloud.tigiminsight.com/index.php/s/eNCU70mrTem6WjF)

    $(ros) python dataset_preparation.py

And then, the model is ready to train. The trained model and the numpy array of labels used here are in the [link](https://cloud.tigiminsight.com/index.php/s/CaA4I9rUDklID2F)    

    $(ros) python pos_predictor_training.py


### > Evaluation  

Training takes a little long time although it totally relies on hardware. After training, there are two files generated in /pos_predicition_model, one for labels called '**autodock_pos_labels.npy**', other one for model weights named '**mobilenetv2_weights_XXX.h5**' where XXX is epochs. With current hyper parameters, the validation loss is almost 0.50 and validation accuracy is around 0.9 which shows almost 100% acurrate in choosing the right one out of 1464 classes.   

Please make sure that the Gazebo simulation is still run. Optionally, turtlebo3_teleop can be used to control manually, but not recommended.  

    # press 'n' or 'r' if you want to change the position of docking station
    $(ros) python model_eval.py     

Here is a [video](https://www.youtube.com/watch?v=olI7jhhOlT8). 

    # able to evaluate the model with validata dataset. No Gazebo simulation is needed. 
    $(ros) python model_eval.py --run dataset

    # able to evaluate the model with real images taken by a real Raspberry Pi camera.  No Gazebo simulation is needed.
    # put an image in the sample_images folder. 
    $(ros) python model_eval.py --run images 

![](./(0.3_0.4_-0.1)_training_images_5_real_image_1.png)  
The above image show that the model is able to use in real environment, because the bottom-middle in the image is captured by iPhone camera and rescaled to (100x100). The accuracy is 50% which is not good, but it could be better if the model is trained with more dataset including real images.  

## 5.Move to (0.0 , 1.0)  
  
Once the position predictor is trained well, the bot can estimate where it is. Then it can move to 'ready-to-dock' location where the turtlebot3 faces the docking station with appropriate distance from it so that it could reach the  station successfully.  

This is a simple quest to solve only using legacy approach, because the bot already knows its position and yaw. There is nothing to do it here, because this state is automatically tiriggered when the pos prediction model works properly.  


## 6.Reinforcement Learning based Motion Control  

Although the robot is anyhow able to predict where it is, the position is estimated value, not correct one. Therefore, reinforcement learning has been introduced so as to guide it to the docking station very accurately 

The main RL algorithm is PPO, and a capture image takes in as input, the RL agent predicts both angular and lienar velocities to get to the target.

### > Training RL   
Like other deep learning technologies, training is the first step to do before using.   

    # Please make sure that Gazebo simulation is running on the other terminal 
    $(ros) python rl_motion_control.py --mode "train"



## Reference  
[ROS Material List](http://wiki.ros.org/simulator_gazebo/Tutorials/ListOfMaterials) - deprecated  