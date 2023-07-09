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
from Agent_ddpg import Agent
from OUActionNoise import OUActionNoise
from env import CustomEnvironment
from utils_ddpg import manage_memory

if __name__ == '__main__':

    env = CustomEnvironment()

    evaluate=True
    agent = Agent(input_dims=[2], env=env,
                  alpha=0.001, beta=0.01)
    agent.load_models()
    count=0
    for i in range(100):
        state = env.reset()
        done = False
        score = 0
        n_steps=0
        dist_=4000
        acceleration_missile=[]
        while not done and n_steps<6000:

            action = agent.choose_action(state, True)
            acceleration_missile.append(action*400)
            action=action.numpy()
            state_,reward,done,dist=env.step(action[0]*400,n_steps,True)
            state=state_
            n_steps+=1

        print(done)

        if done:
          count=count+1
          points_m,points_t,acce=env.pos_history()
          start_object1 = points_m[0]
          start_object2 = points_t[0]
          start_object3 = points_m[-1]


          # Plot the starting points
          plt.plot(start_object1[0], start_object1[1], marker='s',color='purple', markersize=10, label='Missile Base')
          plt.plot(start_object2[0], start_object2[1], marker='o',color='red', markersize=10, label='Target Base')
          plt.plot(start_object3[0], start_object3[1], marker='*',color='green', markersize=10, label='Collision Point')

          plt.plot(points_m[:, 0], points_m[:, 1], '-',color='purple')
          plt.plot(points_t[:, 0], points_t[:, 1], '-',color='red')

          plt.xlabel('meters(m)')
          plt.ylabel('meters(m)')
          plt.title('Trajectory Curve')
          plt.legend()
          plt.show()
          plt.plot(acceleration_missile)
          plt.xlabel('Time(s)')
          plt.ylabel('Action(m/s^2)')
          plt.show()
print(count)
