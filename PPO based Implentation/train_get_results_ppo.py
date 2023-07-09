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
from ppo_agent import Agent
from env import CustomEnvironment

EPOCHS = 1000
MAX_EPISODE_STEPS = 6000
# train at the end of each epoch for simplicity. Not necessarily better.
TRAJECTORY_BUFFER_SIZE = MAX_EPISODE_STEPS
BATCH_SIZE = 128
RENDER_EVERY = 100


if __name__ == "__main__":
    env = CustomEnvironment()
    agent = Agent(1, (2,),
                  BATCH_SIZE, TRAJECTORY_BUFFER_SIZE)
    for epoch in range(EPOCHS):
        s = env.reset()
        r_sum = 0
        for t in range(MAX_EPISODE_STEPS):

            # get action from agent given state
            s=np.array(s)
            a = agent.choose_action(s)
            # get s_,r,done
            a_t=max(-1,min(1,a[0]))
            # print(a_t)
            s_, r, done, _ = env.step(a_t*400,t)

            agent.store_transition(s,a, r)
            r_sum += r
            if done or (t == MAX_EPISODE_STEPS-1):
                # predict critic for s_ (value of s_)
                s_=np.array(s_)
                last_val = r if done else agent.get_v(s_)
                # do the discounted_rewards and GAE calucaltions
                agent.memory.finish_path(last_val)
                break
        # sometimes render
        if epoch % RENDER_EVERY == 0:
            env.render()
        agent.train_network()
        agent.memory.clear()

        print(f"Episode:{epoch}, step:{t}, r_sum:{r_sum}, done{done}")


