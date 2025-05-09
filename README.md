# DIA Coursework (DIA-CW)

This repository contains code for the DIA coursework, including two main components: A* Algorithm and Reinforcement Learning (RL) using PPO and A2C algorithms. The repository provides implementations for solving environment-based tasks, including search-based pathfinding and training RL agents for warehouse navigation using MiniGrid.

## Folder Structure

- `astar/`: Contains the implementation of the A* Algorithm for pathfinding in different environments.
- `reinforcement_learning/`: Contains scripts and models for training and evaluating reinforcement learning agents using PPO and A2C algorithms.

## A* Algorithm

### Overview
This folder contains the implementation of the A* search algorithm. The algorithm is used to find the optimal path in various environments.

### How to Run A star algorithm
1. Navigate to the `astar/` folder.
2. Run the following command to execute the A* search:
   ```bash
   python astarmain.py

### How to Run Reinforment learning - AC and PPO algorithm
1. Navigate to the `astar/` folder.
2. Run the following command to execute the A* search:
   ```bash
   python astarmain.py

## How to Train
Navigate to the `reinforcement_learning/` folder.
To train a model, you can choose between the PPO and A2C algorithms. Use the following command:

python -m scripts.train --algo a2c --env MiniGrid-Warehouse --layout 2 --model modela2c1 --save-interval 10 --frames 1000000 for example
- Model, algo (a2c, ppo), frames can be changed

##### --algo: Choose between ppo or a2c (the algorithm to use).
##### --env: The environment to train in (e.g., MiniGrid-Warehouse).
##### --layout: The layout of the environment (e.g., 2 for a specific layout).
##### --model: The name of the model file to save.
##### --save-interval: The interval at which to save the model.
##### --frames: The number of frames to train the agent.

python -m scripts.evaluate --env MiniGrid-Warehouse --layout 2 --model modela2c1 for example
- Model only can be changed

python -m scripts.visualize --env MiniGrid-Warehouse --layout 2 --model modela2c1 for example

## TensorBoard
For training analysis and to view the training progress, run TensorBoard:

Navigate to the reinforcement_learning folder and run the following command:

tensorboard --logdir=./storage/
Open the provided link in your browser to view the training graphs and result

## Implementation Details
The models are implemented using the Torch-ac library for training reinforcement learning agents.

We experimented with various hyperparameters and settings to optimize agent performance in the environment.



