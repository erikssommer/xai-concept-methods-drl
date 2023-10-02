import numpy as np
from utils import config
from env import gogame
from policy import ActorCriticNet

class Agent:
    def __init__(self, path, name):
        self.name = name # Naming the player the same as the network for clarity

        self.player_1_win = 0
        self.player_2_win = 0

        self.player_1_loss = 0
        self.player_2_loss = 0

        self.win = 0
        self.loss = 0
        self.draw = 0
        self.nn = ActorCriticNet(config.board_size, (f'{path}/{name}'))

    # Play a round of the turnament
    def choose_action(self, state):
        distribution, _ = self.nn.predict(state)
        action = np.argmax(distribution[0])
        # Test if move is valid
        valid_moves = gogame.valid_moves(state)

        if valid_moves[action] != 1:
            print("Invalid move, choosing random move")
            valid_move_idcs = np.argwhere(valid_moves).flatten()
            return np.random.choice(valid_move_idcs), False
        
        return action, True

    # Add a win
    def add_win(self, player):
        self.win += 1

        if player == 1:
            self.player_1_win += 1
        else:
            self.player_2_win += 1

    # Add a loss
    def add_loss(self, player):
        self.loss += 1

        if player == 1:
            self.player_1_loss += 1
        else:
            self.player_2_loss += 1

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