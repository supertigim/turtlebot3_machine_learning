## 1.Introduction  
Autonomous Docking Module is implemented using deep learning technology. 

Idea behind this task is that 3 three steps are required to do. 
1. Figure out where the turtlbebot3 is
    - Predict the position relative to the docking spot by using an image captured on top of the bot 
2. Situate the bot one meter way from the spot and face it with exact angle. 
    - Move the robot to (0, 1) pose, where pose of the spot is (0, 0) 
3. Try to reach the spot without angle problem 
    - Use reinforcement learning to get to the spot (Input: an image, Output: angular/linear velocities)

## 2.Prerequisites   

1.[ROS kinectic on ubunt 16.04](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
2.[Turtlebot3 Dependant ROS Packages](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
3.[Turtlebot3 Simulation ROS Package](http://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#simulation)  
4.OpenAI baselines for reinforcement learning  

## 3.Dataset Preparation for pose prediction    

Posistion predictor model needs to be trained by dataset which is pairs of an image and its position label.
Dataset are stored in 'dataset' folder where many folders named with position information are created and capture images belong to a dedicated folder automatically.  
  
First open two terminals,  

### > First terminal  
    # in case of using anaconda  
    source activate ros             # python2.7, pip install rosinstall msgpack empy defusedxml netifaces  

    # Create a gazebo environment
    export TURTLEBOT3_MODEL=burger 
    roslaunch turtlebot3_gazebo turtlebot3_docking_station.launch

#### > Sencond terminal  
    # in case of using anaconda  
    source activate ros             # python2.7, pip install rosinstall msgpack empy defusedxml netifaces  

    # Capture an image while moving the bot randomly 
    cd /home/your_id/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_auto_docking/src  
    python generate_dataset.py  

### > Environment Change 

Virtual environment in Gazebo is quite limited, so try to change it as many as possible while training. 
Here are two ways you can do.  

1. Put light on gazebo 
2. Change material of walls by editing turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_small_square/model.sdf  
    - Material list can be found in /usr/share/gazebo-7/media/materials/scripts/gazebo.material  
    - Find available one by keyword "material Gazebo/"  

## 4.Position Predictor Neural Network Training  

### > Triaining and Test dataset separation  

There are two dataset required in order to train neural networks properly. Training dataset is just for training as guessed by its name, meanwhile test dataset is to evaluate the model to check if it is trained well.  

    python pos_predictor_training.py

## 5.Move to (0.0 , 1.0)  
  
Once the position predictor model is trained well, the bot can estimate where it is. Then it can move to 'ready-to-dock' location where the turtlebot3 faces the docking station with appropriate distance from it so that it can reach the  station successfully.  

This is a simple quest to solve only with legacy approach, because the bot already knows its position and heading.  

    python execute_auto_dock_approach.py --step "step2"  


## 6.Reinforcement Learning based Motion Control  

Although the robot is able to somehow predict where it is, the position is estimated value, not correct one. Therefore, reinforcement learning has been introduced so as to guide it to the docking very accurately from (0.0, 1.0).  

    python rl_motion_control.py --mode "train"





## Reference  
[ROS Material List](http://wiki.ros.org/simulator_gazebo/Tutorials/ListOfMaterials) - deprecated  