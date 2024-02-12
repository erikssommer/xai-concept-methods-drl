from policy import ConvNet
from policy import ResNet
import numpy as np
from typing import Tuple


class Agent:
    def __init__(self,
                 board_size: int,
                 path: str,
                 name: str,
                 greedy_move: bool = False,
                 resnet: bool = False):

        self.name = name  # Naming the player the same as the network for clarity
        self.greedy_move = greedy_move

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
        if resnet:
            self.nn = ResNet(board_size, (f'{path}/{name}'))
        else:
            self.nn = ConvNet(board_size, (f'{path}/{name}'))

    # Play a round of the turnament
    def choose_action(self, state: np.ndarray, valid_moves: np.ndarray) -> Tuple[int, float]:
        return self.nn.best_action(state, valid_moves, self.greedy_move)

    # Add a win
    def add_win(self, player: int) -> None:
        self.win += 1

        if player == 1:
            self.black_win += 1
        else:
            self.white_win += 1

    # Add a loss
    def add_loss(self, player: int) -> None:
        self.loss += 1

        if player == 1:
            self.black_loss += 1
        else:
            self.white_loss += 1

    # Add a draw
    def add_draw(self) -> None:
        self.draw += 1

    # Reset the agent's score
    def reset_score(self) -> None:
        self.score = 0

    # Get the agent's score
    def get_score(self) -> int:
        return self.score

    # Get the agent's name
    def get_name(self) -> str:
        return self.name

    def save_model(self, path) -> None:
        self.nn.model.save(path + self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name
