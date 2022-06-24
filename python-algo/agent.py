from array import array
from pickle import TRUE
from tkinter import scrolledtext
import torch
import numpy as np
from collections import deque
import random
from algo_strategy import
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # Why not increase this?
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Controls the randomness of the agent
        self.gamma = 0.9  # Discount rate
        # Call popleft() if memory is exceeded
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3) #change these values to your game. u can also change the hidden size. what about adding an additional hidden layer?
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: Model, Trainer

    def get_state(self, game):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self):
        self.epsilon = number of games - self.n_games
        final move = the action array
        if random.randint(0,200) < self.epsilon:
            move.random.randint(0, and the next number
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() make sure this works for your specific logic... maybe instead of this just do absolute value of 1
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game =
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        

        # Train short memory of the agent
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember and store in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory and plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save()
            
            print("Game", agent.n_games, "Score", score, "Record", record_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
