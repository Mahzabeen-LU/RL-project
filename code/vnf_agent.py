import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import TensorBoard
import keras
from collections import deque
import numpy as np
import random

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 256
MODEL_NAME = "Test model"
TARGET_UPDATE_FREQUENCY = 5

# Agent class
class DQNAgent_V1:
    def __init__(self, env, discount = 0.99, double = True):
        self.discount = discount
        self.env = env

        self.double = double

        # Main model
        self.model = self.create_model()
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # replay buffer
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    #generate the model to replace the q table
    def create_model(self):
        return Dense_Model(self.env.ACTION_SPACE_SIZE)

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        #Start training when replay buffer is full
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #Randomly sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([self.env.get_state_vector(transition[0][0], transition[0][1], step) for transition in minibatch])
        new_states = np.array([self.env.get_state_vector(transition[3][0], transition[3][1], step) for transition in minibatch])

        model_q = self.model.predict(current_states)
        model_newq = self.model.predict(new_states)

        target_model_newq = self.target_model.predict(new_states)

        #x and y for the nn
        X = []
        y = []

        #Enumerate minibatches
        for idx, (current_state, action, reward, new_state, done) in enumerate(minibatch):

            if self.double:
                #https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn
                #double deep q learning approach   
                #yj = rt+1 + discount * max(Q(state', a, w'))
                if not done:
                    max_future_q_index = np.argmax(model_newq[idx])
                    q_value = target_model_newq[idx][max_future_q_index]
                    new_q = reward + self.discount * q_value
                #if it is the terminal state new_q is the reward
                else:
                    new_q = reward

            else:
                #deep q learning approach   
                #yj = rt+1 + discount * max(Q(state', a, w'))
                if not done:
                    max_future_q = np.max(target_model_newq)
                    new_q = reward + self.discount * max_future_q
                #if it is the terminal state new_q is the reward
                else:
                    new_q = reward

            # Update q value for given state
            current_qs = model_q[idx]
            current_qs[action] = new_q

            X.append(self.env.get_state_vector(current_state[0], current_state[1], step))
            y.append(current_qs)

        # Fit on all samples as one batch
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > TARGET_UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Get q value with a prediction
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

class Dense_Model(tf.keras.Model):

  def __init__(self, num_actions):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dropout1 = keras.layers.Dropout(0.1)
    self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dropout2 = keras.layers.Dropout(0.1)
    self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dropout3 = keras.layers.Dropout(0.1)
    self.dense3 = tf.keras.layers.Dense(num_actions, activation=tf.nn.softmax)

  def call(self, input):
    x = self.dense1(input)
    x = self.dropout1(x)
    x = self.dense2(x)
    x = self.dropout2(x)
    return self.dense3(x)