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

        self.buffer.append((game_state, distribution))
    
    def clear(self):
        self.buffer = deque([], maxlen=self.max_size)