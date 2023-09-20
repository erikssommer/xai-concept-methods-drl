from collections import deque
import random
import numpy as np
from utils.read_config import config
from game.data import GoVars, GoGame

class RBUF:
    """
    Replay buffer for storing training cases for neural network
    """
    def __init__(self, max_size=256):
        self.buffer = deque([], maxlen=max_size)
        self.max_size = max_size

    def get(self, batch_size):
        if batch_size > len(self.buffer):
            # Print info about the buffer
            buffer = self.buffer
            # empty the buffer
            self.buffer = deque([], maxlen=self.max_size)
            return buffer

        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=batch_size)

    def add_case(self, case):
        player, game_state, distribution = case

        self.buffer.append((player, game_state, distribution))

    def set_values(self, winner):
        for i in range(len(self.buffer)):
            player, game_state, distribution = self.buffer[i]
            if winner == GoVars.BLACK and player == GoVars.BLACK:
                value = 1
            elif winner == -1:
                value = 0
            else:
                value = -1
            self.buffer[i] = (game_state, distribution, value)
    
    def clear(self):
        self.buffer = deque([], maxlen=self.max_size)