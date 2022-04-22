# RL-project
This a RL Project fo CISC 856

The environment is created using gym.
To set up the environment run 'run pip install -e .'  on the outermost vnf_env folder

Required Libraries:
gym.py
numpy
matplotlib
tensorflow
keras
gurobipy

python 3.9

vnf_agent.py holds the nueral network and reinforcement learning agent
gurobi_model.py si the gurobi model for benchmarking
Q_learning_main.py is the main reinfrocement learning loop. Run this file to run to train a model.