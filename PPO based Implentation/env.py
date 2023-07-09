import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random

class CustomEnvironment(object):
    def __init__(self, num_states=2, num_actions=20, action_low=-400, action_high=400, taum=0.2, vm=700, vt=250, gamma_t=0,
                 a_v_max=400, a_t_max=100,input_shape=[2]):
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_low = action_low
        self.action_high = action_high
        self.taum = taum
        self.vm = vm
        self.vt =0
        self.gamma_t = gamma_t
        self.a_v_max = a_v_max
        self.a_t_max = a_t_max
        self.ka = -0.2
        self.kr = -2.0
        self.kdr = -2.0
        self.kter = 10.0
        self.r0 =0
        self.lambdaa = 0
        self.gamma_m =0
        self.a_m =0
        self.a_t=0
        self.state = (0,0)
        self.action = 0
        self.r_prime = 0
        self.lambdaa_prime =0
        self.input_shape=input_shape

    def reset(self):


        pos_m_x, pos_m_y = [0, 0]

        self.r=5000
        self.r0 = self.r
        self.lambdaa = math.pi/18
        # self.lambdaa=generate_random_value(0,2*math.pi)
        pos_t_x, pos_t_y =[self.r*np.cos(self.lambdaa),self.r*np.sin(self.lambdaa)]
        # self.lambdaa=math.atan((pos_t_y-pos_m_y)/(pos_t_x-pos_m_x))
        self.gamma_m = math.pi/9
        self.gamma_t = math.pi-math.pi/9

        min_value = -100.0
        max_value = 100.0

        square_count = 30
        square_duration = 200

        self.acce_tar = np.random.uniform(min_value, max_value, size=(square_count, 1))
        self.acce_tar = np.tile(self.acce_tar, (1, square_duration)).flatten()
        self.a_t = self.acce_tar[0]
        self.a_t=0

        self.values = [self.r, self.lambdaa, self.gamma_m, self.a_m, self.gamma_t]

        self.r_prime = self.vt * np.cos(self.gamma_t - self.lambdaa) - self.vm * np.cos(self.gamma_m - self.lambdaa)
        self.lambdaa_prime = (self.vt * np.sin(self.gamma_t - self.lambdaa) - self.vm * np.sin(self.gamma_m - self.lambdaa)) / self.r

        self.r0_prime=self.r_prime
        self.lambdaa0_prime=self.lambdaa_prime

        self.gamma_m_prime = self.a_m / self.vm
        self.gamma_t_prime = 0 if self.vt==0 else self.a_t/self.vt
        self.a_m_prime = (self.action - self.a_m) / self.taum

        values_prime=[self.r_prime, self.lambdaa_prime, self.gamma_m_prime, self.a_m_prime, self.gamma_t_prime]

        self.state = (1.0,1.0 )

        self.points_m = np.empty((0, 2), dtype=float)
        self.points_t= np.empty((0, 2), dtype=float)
        self.point_m=np.array([[pos_m_x,pos_m_y]])
        self.point_t=np.array([[pos_t_x,pos_t_y]])

        self.points_m = np.append(self.points_m, self.point_m, axis=0)
        self.points_t = np.append(self.points_t, self.point_t, axis=0)

        return self.state

    def step(self, action, step_num, evaluate=True):

      prev_state = self.state
      self.action = action
      self.a_t=self.acce_tar[step_num]
      self.a_t=0

      self.r_prime = self.vt * np.cos(self.gamma_t - self.lambdaa) - self.vm * np.cos(self.gamma_m - self.lambdaa)
      self.lambdaa_prime = (self.vt * np.sin(self.gamma_t - self.lambdaa) - self.vm * np.sin(self.gamma_m - self.lambdaa)) / self.r
      self.gamma_m_prime = self.a_m / self.vm
      self.gamma_t_prime=0 if self.vt==0 else self.a_t/self.vt
      self.a_m_prime = (self.action - self.a_m) / self.taum

      self.values_prime=[self.r_prime, self.lambdaa_prime, self.gamma_m_prime, self.a_m_prime, self.gamma_t_prime]

      self.values=self.values+0.01*np.array(self.values_prime)
      [self.r, self.lambdaa, self.gamma_m, self.a_m, self.gamma_t]=self.values


      self.r_prime = self.vt * np.cos(self.gamma_t - self.lambdaa) - self.vm * np.cos(self.gamma_m - self.lambdaa)
      self.lambdaa_prime = (self.vt * np.sin(self.gamma_t - self.lambdaa) - self.vm * np.sin(self.gamma_m - self.lambdaa)) / self.r
      self.gamma_m_prime = self.a_m / self.vm
      self.gamma_t_prime=0 if self.vt==0 else self.a_t/self.vt
      self.a_m_prime = (self.action - self.a_m) / self.taum

      self.values_prime=[self.r_prime, self.lambdaa_prime, self.gamma_m_prime, self.a_m_prime, self.gamma_t_prime]

      self.state = (self.r_prime/self.r0_prime, self.lambdaa_prime/self.lambdaa0_prime)

      zem = self.r / np.sqrt(1 + (self.r_prime / (self.r * self.lambdaa_prime)) ** 2)

      reward_a = self.ka * ((self.a_m / 400) ** 2)
      reward_z = self.kr * ((zem / self.r0) ** 2)
      reward_dr = self.kdr if self.r_prime >= 0 else 0

      done = (abs(self.r)<=1)

      reward_ter = self.kter if done else 0

      reward = (reward_a + reward_dr+ reward_ter )

      if evaluate:
        self.point_m=self.point_m+np.array([[self.vm*np.cos(self.gamma_m)*0.01,self.vm*np.sin(self.gamma_m)*0.01]])
        self.point_t=self.point_t+np.array([[self.vt*np.cos(self.gamma_t)*0.01,self.vt*np.sin(self.gamma_t)*0.01]])

        self.points_m = np.append(self.points_m, self.point_m, axis=0)
        self.points_t = np.append(self.points_t, self.point_t, axis=0)

      return self.state, reward, done, self.r

    def pos_history(self):
      return self.points_m,self.points_t,self.acce_tar[:len(self.points_m)]
    def render(self):
        start_object1 = self.points_m[0]
        start_object2 = self.points_t[0]

        # Plot the starting points
        plt.plot(start_object1[0], start_object1[1], 'r*', markersize=10, label='Start Object 1')
        plt.plot(start_object2[0], start_object2[1], 'rs', markersize=10, label='Start Object 2')

        plt.plot(self.points_m[:, 0], self.points_m[:, 1], '-',color='red')
        plt.plot(self.points_t[:, 0], self.points_t[:, 1], '-',color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Plane Points')
        plt.show()


