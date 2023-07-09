import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import math
import random
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ReplayBuffer_ddpg import ReplayBuffer
from ActorNetwork_ddpg import ActorNetwork
from CriticNetwork_ddpg import CriticNetwork


class Agent:
    def __init__(self, n_actions=1,n_actions_c=1, input_dims=[2], alpha=0.01, beta=0.01, env=None,
                 gamma=0.99, max_size=500000, tau=0.1,
                 fc1_actor=30, fc2_actor=20,fc1_critic=40,fc2_critic=30, batch_size=64,
                 chkpt_dir='models/ddpg/'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_actions=n_actions_c
#         self.noise = noise
        self.max_action = 400
        self.min_action = -400
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions=n_actions,
                                  fc1_dims=fc1_actor, fc2_dims=fc2_actor)
        self.critic = CriticNetwork(n_actions=n_actions_c,
                                    fc1_dims=fc1_critic, fc2_dims=fc2_critic)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         fc1_dims=fc1_actor, fc2_dims=fc2_actor)
        self.target_critic = CriticNetwork(n_actions=n_actions_c,
                                           fc1_dims=fc1_critic, fc2_dims=fc2_critic)


        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))


        self.tau=0
        self.update_network_parameters()
        self.tau=0.1

    def update_network_parameters(self):
        tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self,add='prj11137'):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir+'actor'+add)
        self.target_actor.save(self.chkpt_dir+'target_actor'+add)
        self.critic.save(self.chkpt_dir+'critic'+add)
        self.target_critic.save(self.chkpt_dir+'target_critic'+add)

        # self.actor.save('/content/drive/My Drive/my_model.h5'+'actor'+add)
        # self.target_actor.save('/content/drive/My Drive/my_model.h5'+'target_actor'+add)
        # self.critic.save('/content/drive/My Drive/my_model.h5'+'critic'+add)
        # self.target_critic.save('/content/drive/My Drive/my_model.h5'+'target_critic'+add)


    def load_models(self,add='prj11137'):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir+'actor'+add)

        self.target_actor = \
            keras.models.load_model(self.chkpt_dir+'target_actor'+add)
        self.critic = keras.models.load_model(self.chkpt_dir+'critic'+add)
        self.target_critic = \
            keras.models.load_model(self.chkpt_dir+'target_critic'+add)
        # file_path = r'/content/drive/My Drive/my_model.h5'
        # self.actor=keras.models.load_model(file_path+'actor'+add)
        # self.target_actor=keras.models.load_model(file_path+'target_actor'+add)
        # self.critic=keras.models.load_model(file_path+'critic'+add)
        # self.target_critic=keras.models.load_model(file_path+'target_critic'+add)

    def choose_action(self, observation, evaluate=False,noise=0.15):
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        actions = self.actor.call(state)

        actions = tf.clip_by_value(actions, -1, 1)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                (states_, target_actions)), 1)
            critic_value = tf.squeeze(self.critic((states, actions)), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
        params = self.critic.trainable_variables
        grads = tape.gradient(critic_loss, params)
        self.critic.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic((states, new_policy_actions))
            actor_loss = tf.math.reduce_mean(actor_loss)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        self.update_network_parameters()