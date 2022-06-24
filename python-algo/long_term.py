# from array import array
# from pickle import TRUE
# from tkinter import scrolledtext
# import torch
# import numpy as np
# from collections import deque
# import random
# # from algo_strategy import
# from model import Linear_QNet, QTrainer
# from helper import plot

# store a variable storing all the game states here, and then do short term and long term training here. also store reward here


class agent_info:

    def __init__(self, enemyHealths, yourHealths, agent_reward, totalStates, totalGames):
        self.enemyHealths = enemyHealths
        self.yourHealths = yourHealths
        self.agent_reward = agent_reward
        self.totalStates = totalStates
        self.totalGames = totalGames
    
       


    def updateEnemyHealth(self, health):
        self.enemyHealths.append(health)

    def updateReward(self, score):
        self.agent_reward += score

    def updateYourHealth(self, health):
        self.enemyHealth.append(health)

    def clearStates(self):
        self.totalStates = []
 
    def gameOver(self):
        self.totalGames+=1
        #long term
        self.clearStates()


    

