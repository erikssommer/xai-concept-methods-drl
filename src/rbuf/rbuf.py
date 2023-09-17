from collections import deque
import random
import numpy as np
from utils.read_config import config

class RBUF:
    """
    Replay buffer for storing training cases for neural network
    """
    def __init__(self, max_size=256):
        self.buffer = deque([], maxlen=max_size)
        self.max_size = max_size

    def get(self, batch_size):
        if batch_size > len(self.buffer):
            return self.buffer

        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=batch_size)

    def add_case(self, case):
        player, game_state, distribution = case

        state = transform(player, game_state)

        if player == 1:
            distribution = np.array(distribution).reshape(config.board_size, config.board_size).T.flatten().tolist()

        self.buffer.append((state, distribution))
    
    def clear(self):
        self.buffer = deque([], maxlen=self.max_size)

def transform(player, game_state):
    """
    Transform game state to a format that can be used by the neural network
    :param player: Player number
    :param game_state: Game state
    :return: Transformed game state
    """
    if player == 0:
        return game_state
    else:
        return np.flip(game_state, (0, 1))