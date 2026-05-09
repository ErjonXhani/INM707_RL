# INM707 - Deep Reinforcement Learning

# Deep-Reinforcement-Learning-CW
Coursework project for INM707: Deep Reinforcement Learning at City St Georges, University of London. The project consists of 2 tasks: 

1) To create an envirnoment, defining its state transition function, setting and running Q-Learning algorithm, and Analyze it quantitatively and qualitatively.
2) To create an advanced envirnoment and implement Deep Q Learning with two improvements, and a Deep Reinforcement Learning Technique each. And then Analysing the results quantitatively and qualitatively.

Environment Inspiration: https://store.steampowered.com/app/250900/The_Binding_of_Isaac_Rebirth/

<img width="1211" height="680" alt="image" src="https://github.com/user-attachments/assets/98f287ce-02a4-4af1-a224-ac839855a22b" />
The above is a screenshot from the game we took inspiration from.

# General Information

## Imports and Requirement

Download all requirements from requirements.txt

## Running the code

1) All code are stored in .py file. Hence the general code to run them is to first navigate the respective directory and run the following: ``` !python [filename].py ```
2) Some codes have specfic arguments to pass, whose instructions are in their respective readme file in the respective folders.


# File Structure

## Game

Contains all the code and assests for a full playable envirnoment. ( We recommend you to run it with reduced volume, there are some issues with the level of audio :))

<img width="1208" height="951" alt="image" src="https://github.com/user-attachments/assets/d66d2c4c-92de-4cea-99b3-366c655d1fb6" />
THe above is a screenshot from our game.

For all the algorithms in Advanced_Task and Individual_Task, we shall not be loading assests like images and video to save computational power for training or inference.

## Base_Task

Contains all the code, results and graphs from the Basic Task covering Tasks 1,2,3,4,5, and 6.

Code contains implementation of Q-Learning

<img width="420" height="300" alt="image" src="https://github.com/user-attachments/assets/b322854d-3dea-4257-a270-2384fd89782b" />
The above is a screenshot of the envirnoment used for the basic task.


## Advanced_Task

Contains all the code, and checkpoints from the Advanced Task covering Task 7 and 8.

Code contains implementaiton of Deep Q Learning, Double Deep Q Learning, and Double Deep Q Learning with Prioritized Experience Replay.


## Individual_Task

Contains all the code, and checkpoints from the Advanced Task covering Task 9 and 10.

Code contains implementation of Proximal Policy Optimization (PPO) by Erjon, and Advantage Actor-Critic (A2C) by Badri.

To run the file on training or inference, follow the same instructions as in the advanced task.
