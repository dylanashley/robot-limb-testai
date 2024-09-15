# Gazebo Simulation

## Setup and Execution
    - Clone this repo (```git clone git@github.com:BabyRobotics/gazebo_simulation.git```)
    - Install Gazebo-sim version Harmonic for Ubuntu 22.04
        - https://gazebosim.org/docs/harmonic/install_ubuntu
    - cd into this directory (```cd gazebo_simulation```)
    - Run the bash script (```source setup.sh```)
    - Change to the models directory
        - ```cd models```
    - Execute the simulation
        - Without visualization
            - ```gz sim -s -v 4 arm_sim.sdf```
        - With visualization
            - ```gz sim -r arm_sim.sdf```  (note: -r starts the simulation running automatically. Remove to start paused.)

## Configuration
    - Look in ```./models/arm_sim.sdf``` at the top to change max simulation run speed
    - Look in ```./models/Arm/model.sdf``` at the bottom to change the plugin configuration (e.g. how to talk to the brain)
