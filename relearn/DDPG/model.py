import argparse
import sys
import os
import numpy as np
import pandas as pd
import time


#Keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
#Custom env
from trade_env import TradeEnvironment
#Custom attention
from attention_class import MultiHeadSelfAttention #https://apoorvnandan.github.io/2020/05/10/transformer-classifier/


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A DDPG reinforcement learning algo for crypto trading. Runs in tensorflow 2.
                                                Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy algorithm for learning continous actions.
                                                It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay
                                                and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces.''')

parser.add_argument('--train_data', nargs=1, type= str, default=sys.stdin, help = 'Path to training data.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS AND CLASSES################
#Add slow policy update so that the policy change from the previous update is not too different
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states[0],num_states[1]))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity,  num_states[0],num_states[1]))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index:index+batch_size] = obs_tuple[0]
        self.action_buffer[index:index+batch_size] = obs_tuple[1]
        self.reward_buffer[index:index+batch_size] = obs_tuple[2]
        self.next_state_buffer[index:index+batch_size] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        #Get and apply gradients
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



#Critic and actor models
#The actor should predict 3 states: buy/sell/hold and the percentage for buy/sell
def get_actor():

    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    inputs = layers.Input(shape=num_states)
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    #Flatten
    out = layers.Flatten()(out)
    buy_sell_remain = layers.Dense(3, activation="softmax")(out)
    how_much = layers.Dense(1, activation="softmax",kernel_initializer=last_init)(out)

    #Get argmax to determine buy_sell_remain
    buy_sell_remain = tf.math.argmax(buy_sell_remain,axis=-1,output_type=tf.dtypes.int32)
    #Float conversion
    buy_sell_remain = tf.cast(buy_sell_remain, tf.dtypes.float32)
    #Reshape to fit concat
    buy_sell_remain = tf.expand_dims(buy_sell_remain,axis=1)
    #Outputs
    outputs = layers.Concatenate()([buy_sell_remain,how_much])
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():
    #Here a transformer should be built - learning the state-acition relationship
    # State as input
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(32, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Flatten()(state_out)
     # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)
    action_out = layers.Flatten()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    #Predict how good the action is in the current state
    outputs = layers.Dense(1)(concat)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

#policy() returns an action sampled from our Actor network plus some noise for exploration.
def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [sampled_actions]


#####################Problem definition#####################
#Parse args
args = parser.parse_args()
#Data
data = pd.read_csv(args.train_data[0])
data = data[['open', 'high', 'low', 'close','volume']].values
#Need to create an env in rl gym
batch_size=32
fee = 0.001
look_back_window = 60*24
prediction_window=60*24
env = TradeEnvironment(data,batch_size, fee, look_back_window,prediction_window)

#States and actions
num_actions = 2
num_states = (data.shape[1]+num_actions,look_back_window)
lower_bound=0
upper_bound=2
num_heads=2
ff_dim=8

#Train
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
#get first set of actor and critic
actor_model = get_actor()
critic_model = get_critic()
print(actor_model.summary())
print(critic_model.summary())
#get second set of actor and critic
target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.0001
actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.9
# Used to update target networks - update slowly
tau = 0.0005 #This parameter is crucial which is why PPO may be better
#Memory size: number of rememebered experiences, batch_size
buffer = Buffer(100000, batch_size)

'''
Now we implement our main training loop, and iterate over episodes.
We sample actions using policy() and train with learn() at each time step,
along with updating the Target networks at a rate tau.
'''

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

for ep in range(total_episodes):
    #Reset env
    prev_state = env.reset()
    done=False
    #Reward during episode
    episodic_reward = np.zeros(batch_size)

    #Go through the num steps and trade for the entire prediction window - one step at a time

    for j in range(prediction_window):
        if done==False:
            # Uncomment this to see the Actor in action
            # env.render()
            tf_prev_state = tf.convert_to_tensor(prev_state)
            #Get action based on state
            action = policy(tf_prev_state, ou_noise)
            # Recieve state and reward from environment.
            state, reward,done = env.step(action[0],j)
            #Save to buffer
            buffer.record((prev_state, action[0], reward, state))
            #Save reward
            episodic_reward = reward[:,0]
            #Train the actor/critic
            buffer.learn()
            #Update the target networks slowly using the new things learned in the
            #gradient opt models
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)
            #Update prev state
            prev_state = state

        else:
            break



    ep_reward_list.append(np.average(episodic_reward))

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-1])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

pdb.set_trace()
