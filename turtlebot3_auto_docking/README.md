## Introduction  
Autonomous Docking Module is implemented using deep learning technology. 

Idea behind this task is that 3 three steps are required to do. 
1. Figure out where the turtlbebot3 is
    - Predict the position relative to the docking spot by using an image captured on top of the bot 
2. Situate the bot one meter way from the spot and face it with exact angle. 
    - Move the robot to (0, 1) pose, where pose of the spot is (0, 0) 
3. Try to reach the spot without angle problem 
    - Use reinforcement learning to get to the spot (Input: an image, Output: angular/linear velocities)

## Prerequisites   

    - [ROS kinectic on ubunt 16.04](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
    - [Turtlebot3 Dependant ROS Packages](http://emanual.robotis.com/docs/en/platform/turtlebot3/pc_setup/#pc-setup)  
    - [Turtlebot3 Simulation ROS Package](http://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#simulation)   
    - OpenAI baselines for reinforcement learning

## Dataset Preparation for pose prediction    
  
    # in case of using anaconda  
    source activate ros             # python2.7, pip install rosinstall msgpack empy defusedxml netifaces  

    export TURTLEBOT3_MODEL=burger  


## Reference  
[ROS Material List](http://wiki.ros.org/simulator_gazebo/Tutorials/ListOfMaterials)