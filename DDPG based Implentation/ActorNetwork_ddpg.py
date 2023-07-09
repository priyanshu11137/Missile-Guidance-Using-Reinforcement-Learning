import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=30, fc2_dims=20, n_actions=1):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.006))
        self.fc2 = Dense(self.fc2_dims, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.006))
        self.mu = Dense(self.n_actions, activation='tanh',kernel_regularizer=tf.keras.regularizers.L2(0.006))

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu