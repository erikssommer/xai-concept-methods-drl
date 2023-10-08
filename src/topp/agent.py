import numpy as np
from utils import config
from env import gogame
from policy import ActorCriticNet
import random

class Agent:
    def __init__(self, path, name):
        self.name = name # Naming the player the same as the network for clarity

        self.player_black = 0
        self.player_white = 0

        self.black_win = 0
        self.white_win = 0

        # Black is allways the starting player
        self.black_loss = 0
        self.white_loss = 0

        self.win = 0
        self.loss = 0
        self.draw = 0
        self.nn = ActorCriticNet(config.board_size, (f'{path}/{name}'))

    # Play a round of the turnament
    def choose_action(self, state):
        return self.nn.best_action(state)

    # Add a win
    def add_win(self, player):
        self.win += 1

        if player == 1:
            self.black_win += 1
        else:
            self.white_win += 1

    # Add a loss
    def add_loss(self, player):
        self.loss += 1

        if player == 1:
            self.black_loss += 1
        else:
            self.white_loss += 1

    # Add a draw
    def add_draw(self):
        self.draw += 1

    # Reset the agent's score
    def reset_score(self):
        self.score = 0

    # Get the agent's score
    def get_score(self):
        return self.score

    # Get the agent's name
    def get_name(self):
        return self.name

    def save_model(self, path):
        self.nn.model.save(path + self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name