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
    best_score=-40000
    avg_score=-40000
    manage_memory()
    env = CustomEnvironment()

    agent = Agent(input_dims=[2], env=env,
                  alpha=0.001, beta=0.001)
    agent.load_models()
    n_episodes = 200
    count=0
    score_history = []
    knoise=OUActionNoise()


    evaluate = False

    for i in range(n_episodes):
        observation = env.reset()
        knoise.reset()
        done = False
        score = 0
        n_steps=0
        dist_=10000
        while not done and n_steps<6000:
            action = agent.choose_action(observation, evaluate)
            action=action.numpy()
            sample=knoise()

            action_p=action[0]+sample
            action_p = max(-1,min(1,action_p))
            action_t=action_p
            action_t*=400


            observation_, reward, done,dist = env.step(action_t,n_steps)
            dist_=min(dist,dist_)

            agent.store_transition(observation,action_p , reward,
                                    observation_, done)
            score=score+reward

            observation = observation_
            n_steps+=1
        agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-30:])
        if avg_score>best_score:
            best_score = avg_score
            agent.save_models()
        print('episode {} score {:.1f} avg score {:.1f}'.format(i, score, avg_score))
        print(done)
        print(dist_)
        if done:
            count+=1





    x = [i+1 for i in range(len(score_history))]
    plt.plot(x, score_history)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.show()
    print(count)