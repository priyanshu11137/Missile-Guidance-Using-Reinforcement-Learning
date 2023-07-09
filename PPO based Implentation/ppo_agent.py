import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
import tensorflow.keras as K
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as Kera
import matplotlib.pyplot as plt
import math
import random
from tensorflow.keras import regularizers
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import os
import time
from collections import deque
from scipy import signal
from ppo_memory import MemoryPPO

# %%
class Agent:
    def __init__(self, action_n, state_dim, TRAINING_BATCH_SIZE, TRAJECTORY_BUFFER_SIZE):
        """This initializes the agent object.
        Main interaction is the choose_action, store transition and train_network.
        The agent only requires the state and action spaces to fuction, other than that it is pretty general
        and should be easy to adapt for other continuous envs.
        To understand what is happening, I recommend to look at the ppo_loss method and the build_actor method first.
        The training method itself is more or less only data preperation for calling the fit functions
        for actor and critic. But critic has a trivial loss, so all the PPO magic is in the ppo_loss function.

        """
        self.action_n = action_n
        self.state_dim = state_dim
        # CONSTANTS
        self.TRAINING_BATCH_SIZE = TRAINING_BATCH_SIZE
        self.TRAJECTORY_BUFFER_SIZE = TRAJECTORY_BUFFER_SIZE
        self.TARGET_UPDATE_ALPHA = 0.1
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.1
        self.NOISE = 0.1  # Exploration noise, for continous action space
        # create actor and critic neural networks
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        # for the loss function, additionally "old" predicitons are required from before the last update.
        # therefore create another networtk. Set weights to be identical for now.
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        # for getting an action (predict), the model requires it's ususal input, but advantage and old_prediction is only used for loss(training). So create dummys for prediction only
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2*self.action_n))
        # our transition memory buffer
        self.memory = MemoryPPO(
            self.state_dim, self.action_n, self.TRAJECTORY_BUFFER_SIZE)

    def ppo_loss(self, advantage, old_prediction):
        """The PPO custom loss.
        For explanation see for example:
        https://youtu.be/WxQfQW48A4A
        https://youtu.be/5P7I-xPq8u8
        Log Probability of  loss: (x-mu)²/2sigma² - log(sqrt(2*PI*sigma²))
        entropy of normal distribution: sqrt(2*PI*e*sigma²)
        params:
            :advantage: advantage, needed to process algorithm
            :old_predictioN: prediction from "old" network, needed to process algorithm
        returns:
            :loss: keras type loss fuction (not a value but a fuction with two parameters y_true, y_pred)
        TODO:
            probs = tf.distributions.Normal(mu,sigma)
            probs.sample #choses action
            probs.prob(action) #probability of action
            https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
        """

        def get_log_probability_density(network_output_prediction, y_true):
            """Sub-function to get the logarithmic probability density.
            expects the prediction (containing mu and sigma) and the true action y_true
            Formula for pdf and log-pdf see https://en.wikipedia.org/wiki/Normal_distribution
            """
            # the actor output contains mu and sigma concatenated. split them. shape is (batches,2xaction_n)
            mu_and_sigma = network_output_prediction
            mu = mu_and_sigma[:, 0:self.action_n]
            sigma = mu_and_sigma[:, self.action_n:]
            variance = K.backend.square(sigma)
            pdf = 1. / K.backend.sqrt(2. * np.pi * variance) * K.backend.exp(-K.backend.square(y_true - mu) / (2. * variance))
            log_pdf = K.backend.log(pdf + K.backend.epsilon())
            return log_pdf

        # refer to Keras custom loss function intro to understand why we define a funciton inside a function.
        # here y_true are the actions taken and y_pred are the predicted prob-distribution(mu,sigma) for each n in acion space
        def loss(y_true, y_pred):
            # First the probability density function.
            log_probability_density_new = get_log_probability_density(y_pred, y_true)
            log_probability_density_old = get_log_probability_density(old_prediction, y_true)
            # Calc ratio and the surrogates
            # ratio = prob / (old_prob + K.epsilon()) #ratio new to old
            ratio = K.backend.exp(log_probability_density_new-log_probability_density_old)
            surrogate1 = ratio * advantage
            clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO, max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate2 = clip_ratio * advantage
            # loss is the mean of the minimum of either of the surrogates
            loss_actor = - K.backend.mean(K.backend.minimum(surrogate1, surrogate2))
            # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
            sigma = y_pred[:, self.action_n:]
            variance = K.backend.square(sigma)
            loss_entropy = self.ENTROPY_LOSS_RATIO * K.backend.mean(-(K.backend.log(2*np.pi*variance)+1) / 2)  # see move37 chap 9.5
            # total bonus is all losses combined. Add MSE-value-loss here as well?
            return loss_actor + loss_entropy
        return loss



    def _build_actor_network(self):
        """builds and returns a compiled keras.model for the actor.
        There are 3 inputs. Only the state is for the pass though the neural net.
        The other two inputs are exclusivly used for the custom loss function (ppo_loss).
        """
        # define inputs. Advantage and old_prediction are required to pass to the ppo_loss funktion
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        advantage = K.layers.Input(shape=(1,), name='advantage_input')
        old_prediction = K.layers.Input(shape=(2*self.action_n,), name='old_prediction_input')
        # define hidden layers
        dense = Dense(30, activation='relu', name='dense1',kernel_regularizer=tf.keras.regularizers.L2(0.006))(state)
        dense = Dense(20, activation='relu', name='dense2',kernel_regularizer=tf.keras.regularizers.L2(0.006))(dense)
        # connect layers. In the continuous case the actions are not probabilities summing up to 1 (softmax)
        # but squshed numbers between -1 and 1 for each action (tanh). This represents the mu of a gaussian
        # distribution
        mu = Dense(self.action_n, activation='tanh',name="actor_output_mu",kernel_regularizer=tf.keras.regularizers.L2(0.006))(dense)
        #mu = 2 * muactor_output_layer_continuous
        # in addtion, we have a second output layer representing the sigma for each action
        sigma = Dense(self.action_n, activation='softplus', name="actor_output_sigma",kernel_regularizer=tf.keras.regularizers.L2(0.006))(dense)
        #sigma = sigma + K.backend.epsilon()
        # concat layers. The alterative would be to have two output heads but this would then require to make a custom
        # keras.function insead of the .compile and .fit routine adding more distraciton
        mu_and_sigma = K.layers.concatenate([mu, sigma])
        # make keras.Model
        actor_network = K.Model(
            inputs=[state, advantage, old_prediction], outputs=mu_and_sigma)
        # compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        actor_network.compile(
            optimizer='adam', loss=self.ppo_loss(advantage, old_prediction))
        # summary and return
        actor_network.summary()
        return actor_network

    def _build_critic_network(self):
        """builds and returns a compiled keras.model for the critic.
        The critic is a simple scalar prediction on the state value(output) given an state(input)
        Loss is simply mse
        """
        # define input layer
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        # define hidden layers
        dense = K.layers.Dense(40, activation='relu', name='dense1',kernel_regularizer=tf.keras.regularizers.L2(0.005))(state)
        dense = K.layers.Dense(30, activation='relu', name='dense2',kernel_regularizer=tf.keras.regularizers.L2(0.005))(dense)
        # connect the layers to a 1-dim output: scalar value of the state (= Q value or V(s))
        V = K.layers.Dense(1, name="actor_output_layer",kernel_regularizer=tf.keras.regularizers.L2(0.005))(dense)
        # make keras.Model
        critic_network = K.Model(inputs=state, outputs=V)
        # compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        critic_network.compile(optimizer='Adam', loss='mean_squared_error')
        # summary and return
        critic_network.summary()
        return critic_network

    def update_tartget_network(self):
        """Softupdate of the target network.
        In ppo, the updates of the
        """
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor_network.get_weights(),dtype=object)
        actor_tartget_weights = np.array(self.actor_old_network.get_weights(),dtype=object)
        new_weights = alpha*actor_weights + (1-alpha)*actor_tartget_weights
        self.actor_old_network.set_weights(new_weights)

    def choose_action(self, state, optimal=False):
        """chooses an action within the action space given a state.
        The action is chosen by random with the weightings accoring to the probability
        params:
            :state: np.array of the states with state_dim length
            :optimal: if True, the agent will always give best action for state.
                     This will cause no exploring! --> deactivate for learning, just for evaluation
        """
        assert isinstance(state, np.ndarray)
        assert state.shape == self.state_dim
        # reshape for predict_on_batch which requires 2d-arrays (batches,state_dims) but only one batch
        state = state.reshape(1,-1)
        # the probability list for each action is the output of the actor network given a state
        # output has shape (batchsize,2xaction_n)
        mu_and_sigma = self.actor_network.predict_on_batch(
            [state, self.dummy_advantage, self.dummy_old_prediciton])
        mu = mu_and_sigma[0, 0:self.action_n]
        sigma = mu_and_sigma[0, self.action_n:]
        # action is chosen by random with the weightings accoring to the probability
        if optimal:
            action = mu
        else:
            action = np.random.normal(loc=mu, scale=sigma, size=self.action_n)
        return action
    def train_network(self):
        """Train the actor and critic networks using GAE Algorithm.
        1. Get GAE rewards, s,a,
        2. get "old" prediction (of target network)
        3. fit actor and critic network
        4. soft update target "old" network
        """

        # get randomized mini batches
        states, actions, gae_advantages, discounted_rewards, values = self.memory.get_batch(
            self.TRAINING_BATCH_SIZE)
        gae_advantages = gae_advantages.reshape(-1, 1)  # batches of shape (1,) required
        gae_advantages = K.utils.normalize(gae_advantages)  # optionally normalize
        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(states)

        # Convert tensors to numpy arrays
        batch_old_prediction_np = batch_old_prediction
        batch_old_prediction_np = np.reshape(batch_old_prediction_np, (-1, 2 * self.action_n))

        states_np = states
        states_np = np.reshape(states_np, (-1, self.state_dim[0]))

        gae_advantages_np = gae_advantages
        gae_advantages_np = np.reshape(gae_advantages_np, (-1, 1))

        actions_np = actions
        discounted_rewards_np = discounted_rewards

        # Training the networks
        self.actor_network.fit(
            x=[states_np, gae_advantages_np, batch_old_prediction_np], y=actions_np, verbose=0)
        self.critic_network.fit(
            x=states_np, y=discounted_rewards_np, epochs=1, verbose=0)

        # Soft update the target network
        self.update_tartget_network()



    def store_transition(self, s, a, r):
        """Store the experiences transtions into memory object.
        """
        value = self.get_v(s).flatten()
        self.memory.store(s, a, r, value)

    def get_v(self, state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        s = np.reshape(state, (-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def get_old_prediction(self, state):
        """Makes an prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        state = np.reshape(state, (-1, self.state_dim[0]))
        num_states = state.shape[0]
        old_predictions = []
        for i in range(num_states):
            old_prediction = self.actor_old_network.predict([state[i:i+1], self.dummy_advantage, self.dummy_old_prediciton],verbose=0)
            old_predictions.append(old_prediction)
        return np.concatenate(old_predictions, axis=0)


# %%
