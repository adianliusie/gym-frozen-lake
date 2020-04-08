import gym
import time
import torch
import numpy as np
from random import random
from random import randint

class Class_action_helper():
    def __init__(self, num_actions):
        self.e_start = 1
        self.e_end = 0.1
        self.max_steps = 20000
        self.num_actions = num_actions

    def epsilon(self, time_step):
        if time_step < self.max_steps:
            e = self.e_start + (time_step/self.max_steps)*(self.e_end - self.e_start)
        else:
            e = self.e_end
        return e

    def action_sampler(self, best_decision, time_step):
        if random() < self.epsilon(time_step):
            action = randint(0, 1000)%self.num_actions
        else:
            action = best_decision
        return action


class Q_table_class():
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.Q_table = np.zeros([num_states, num_actions])
        self.gamma = 0.99
        self.alpha = 0.05

    def update_Q_table(self, state, next_state, action, reward):
        next_best = max(self.Q_table[next_state])
        difference = (reward + self.gamma*next_best) - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha*difference

    def random_action(self):
        return randint(0, 1000)%self.num_actions

    def best_action(self, state):
        if not (self.Q_table[state] == 0).all():
            return int(np.argmax(self.Q_table[state]))
        else:
            return self.random_action()

    def best_action_debug(self, state, env, episode):
        action = self.best_action(state)

        if episode > 15000:
            print("debugging the best action_function")
            print(self.Q_table[state])
            print(action)
            print("finished debugging the best action_function")
            env.render()
            time.sleep(0.1)

        return action



def game():
    env = gym.make('FrozenLake8x8-v0')
    observation = env.reset()

    for step in range(100):
        action = int(input("0=left, 1=down, 2=right, 3=up: "))
        new_observation, reward, done, info = env.step(action)
        env.render()

        if done:
            break

def train(env, Q):
    action_maker = Class_action_helper(env.action_space.n)
    success = 0

    for episode in range(20000):
        observation = env.reset()
        score = 0
        for step in range(100):
            best_action = Q.best_action(observation)
            action = action_maker.action_sampler(best_action, episode)
            new_observation, reward, done, info = env.step(action)

            Q.update_Q_table(observation, new_observation, action, reward)

            if reward:
                print(episode)
                if episode>10000:
                    success += 1
            if done:
                break

            observation = new_observation

    print(success/20000)

def eval(env, Q):
    success = 0
    total_walk = 0
    for episode in range(1000):
        observation = env.reset()
        score = 0

        for step in range(100):
            #env.render()
            best_action = Q.best_action(observation)
            new_observation, reward, done, info = env.step(best_action)

            if reward:
                success += 1
                total_walk += step

            if done:
                break

            observation = new_observation
            #time.sleep(0.1)

    print(total_walk/success)
    print(success/1000)

def main():
    env = gym.make('FrozenLake8x8-v0')
    Q = Q_table_class(64, env.action_space.n)

    train(env, Q)
    eval(env, Q)
    eval(env, Q)
    eval(env, Q)

    env.close()

if __name__ == '__main__':
    main()
