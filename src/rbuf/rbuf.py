from collections import deque
import random
from env import govars

class RBUF:
    """
    Replay buffer for storing training cases for neural network
    """
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.buffer = deque([], maxlen=max_size)

        # Lists for storing the training cases of later adition to the buffer
        self.player = []
        self.states = []
        self.distributions = []

    def add_case(self, player, state, distribution):
        self.player.append(player)
        self.states.append(state)
        self.distributions.append(distribution)

    def clear_lists(self):
        self.states = []
        self.player = []
        self.distributions = []

    def get(self, batch_size):
        if batch_size > self.__len__():
            batch_size = self.__len__()

        # Random sample from the buffer
        res = random.sample(self.buffer, batch_size)

        return res

    def set_values(self, winner):
        for (dist, state, player) in zip(self.distributions, self.states, self.player):
            # Get the winning value
            value = self.winning(player, winner)

            # Add the case to the buffer
            # we push a new case to memory, if max size is reached we pop left
            if len(self.buffer) >= self.max_size:
                self.buffer.popleft()
            self.buffer.append((state, dist, value))
        
        self.clear_lists()

    def winning(self, player, winner):
        if player == govars.BLACK and winner == 1:
            return 1
        elif player == govars.WHITE and winner == -1:
            return 1
        elif player == govars.BLACK and winner == -1:
            return -1
        elif player == govars.WHITE and winner == 1:
            return -1
        else:
            AssertionError("Invalid winner")

    def __len__(self):
        return len(self.buffer)