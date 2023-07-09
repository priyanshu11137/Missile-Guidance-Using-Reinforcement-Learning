import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=40, fc2_dims=30,n_actions=1 ):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.005))
        self.fc2 = Dense(self.fc2_dims, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.005),)
        self.q = Dense(self.n_actions, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.005),)

    # have to define inputs as a tuple because the model.save() function
    # trips an error when trying to save a call function with two inputs.
    def call(self, inputs):
        state, action = inputs
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q