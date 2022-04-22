import gym
import numpy as np
import matplotlib.pyplot as plt
from vnf_agent import *
import vnf_env
import random
import math 
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

#SETTINGS
EPISODES = 500
DISCOUNT = 0.99

#EPSILON SETTINGS
epsilon = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001
UPDATE_EVERY = 100

#Initialize environment
env = gym.make("vnf-v0")
random.seed(1)
np.random.seed(1)
env.seed(1)

agent = DQNAgent_V1(env, discount=DISCOUNT, double=True)

#keep track of best solution
best_rewards = -math.inf
best_solution = None

rewards = []

#for each episode
for episode in range(1, EPISODES + 1):

    if episode % 1 == 0:
        print(f"NEW EPISODE: {episode}")
        print(f"epsilon: {epsilon}")
        print()

    #initialize variables
    episode_reward = 0
    step = 0

    # Reset environment and get initial state
    current_state = env.reset()

    # print(current_state)

    # Reset flag and start iterating until episode ends
    done = False

    while not done:

        #used if we only want to select valid actions
        valid = True

        #choose an action
        if np.random.random() > epsilon:
            # Get action from Q table
            qs = agent.get_qs(env.get_state_vector(*current_state)).copy()
            print(qs)
            vnf = env.vnfs[env.curr_step]
        
            #uncomment to only select valid actions
            #may not be able to find a solution unless no actions can be taken
            while True:
                # #if no valid actions
                # if all(x == 0 for x in qs):
                #     print("inside")
                #     valid = False
                #     break
                action = np.argmax(qs)
                # if not episode % UPDATE_EVERY:
                #     print(f"Exploitation {action}")
                # #decrease hosts capacity by vnf_require
                # capacity = env.hosts[action][0] - vnf[3]
                # if capacity < 0:
                #     qs[action] = 0
                #     continue
                # #decrease hosts link by vnf_require
                # link = env.hosts[action][1] - vnf[2]
                # if link < 0:
                #     qs[action] = 0
                #     continue
                break

            if not episode % UPDATE_EVERY:
                print(f"Exploitation {action}")

        else:
            # Get valid action
            valid_actions = list(range(env.ACTION_SPACE_SIZE))
            # vnf = env.vnfs[env.curr_step]

            while True:
                # if not valid_actions:
                #     valid = False
                #     break
                action = random.choice(valid_actions)
                # if not episode % UPDATE_EVERY:
                #     print(f"Exploration {action}")
                # #decrease hosts capacity by vnf_require
                # capacity = env.hosts[action][0] - vnf[3]
                # if capacity < 0:
                #     valid_actions.remove(action)
                #     continue
                # #decrease hosts link by vnf_require
                # link = env.hosts[action][1] - vnf[2]
                # if link < 0:
                #     valid_actions.remove(action)
                #     continue
                break

            if not episode % UPDATE_EVERY:
                print(f"Exploration {action}")


        if not episode % UPDATE_EVERY:
            print("done")        

        if valid:
            new_state, reward, done, _ = env.step(action)

            if reward > best_rewards:
                print(f"New best solution")
                print(reward)
                best_rewards = reward
                best_solution = new_state

            #add reward
            episode_reward += reward

            #update the replay memory and attempt to train the agent
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            #take next step
            current_state = new_state
            step += 1

        #if there is no valid step just move on
        else:
            print("invalid")
            done = env.check_if_done()
            env.increment_step()
            step += 1
            env.invalid_placement()

        #Every UPDATE_EVERY steps when done print the current results
        #used for debugging and ensuring everything is working
        if done and not episode % UPDATE_EVERY:
            print(new_state)
            print(reward)
            print()

    # print(new_state)
    # print()

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    #append rewards, can be used for graphing
    rewards.append(env.reward)

#another iteration of the model

# # Iterate over episodes
# for episode in range(1, EPISODES + 1):

#     if episode % 1 == 0:
#         print(f"NEW EPISODE: {episode}")
#         print(f"epsilon: {epsilon}")
#         print()

#     # Restarting episode - reset episode reward and step number
#     episode_reward = 0
#     step = 0

#     # Reset environment and get initial state
#     current_state = env.reset()

#     # Reset flag and start iterating until episode ends
#     done = False

#     while not done:

#         if np.random.random() > epsilon:
#             # Get action from Q table
#             qs = agent.get_qs(env.get_state_vector(*current_state))
#             print(qs)
#             action = np.argmax(qs)

#             if not episode % UPDATE_EVERY:
#                 print(f"Exploitation {action}")
        
#         else:
#             # Get valid action
#             action = random.choice(list(range(env.ACTION_SPACE_SIZE)))

#             if not episode % UPDATE_EVERY:
#                 print(f"Exploration {action}")

#         new_state, reward, done, _ = env.step(action)

#         if reward > best_rewards:
#             print(f"New best solution")
#             print(reward)
#             best_rewards = reward
#             best_solution = new_state

#         # Transform new continous state to new discrete state and count reward
#         episode_reward += reward

#         # Every step we update replay memory and train main network
#         agent.update_replay_memory((current_state, action, reward, new_state, done))
#         agent.train(done, step)

#         current_state = new_state
#         step += 1

#         #Every UPDATE_EVERY steps when done print the current results
#         if done and not episode % UPDATE_EVERY:
#             print(new_state)
#             print(reward)
#             print()

#     # Decay epsilon
#     if epsilon > MIN_EPSILON:
#         epsilon *= EPSILON_DECAY
#         epsilon = max(MIN_EPSILON, epsilon)

#     rewards.append(env.reward)

plt.plot(range(EPISODES), rewards)
plt.show()
