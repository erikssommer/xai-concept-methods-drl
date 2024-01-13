import numpy as np
import math

from env import govars, gogame
from utils import config

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Node:
    """
    A node in MCTS.
    Optimizations based on https://www.moderndescartes.com/essays/deep_dive_mcts/.
    """

    def __init__(self, state, action=None, c=1, parent=None, nondet_plies=8):
        self.state = state
        self.action = action
        self.is_expanded = False
        self.parent = parent # Optional[Node]
        self.children = {} # Dict[action, Node]
        self.player_number = gogame.turn(state)
        self.c = c
        self.nondet_plies = nondet_plies

        self.child_priors = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)
        self.child_total_value = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)
        self.child_number_visits = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)

        valid_moves = gogame.valid_moves(state)
        # Using valid moves as a mask, we can set illegal moves to 0
        self.illegal_moves_mask = np.ones(self.child_priors.shape)
        self.illegal_moves_mask[valid_moves == 0] = 0


    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def select_leaf(self):
        current = self
        while current.is_expanded:
            child_move = current.best_child()
            # This is a (deferred) leaf, have to create it
            if child_move not in current.children:
                current.add_child(child_move, self.child_priors[child_move])
            current = current.children[child_move]
        return current

    def expand(self, child_priors):
        self.expanded = True
        self.child_priors = child_priors

    def add_child(self, move):
        self.children[move] = Node(gogame.next_state(self.state, move), move, parent=self)

    def backup(self, value):
        if self.parent:
            self.number_visits += 1
            self.total_value += value
            self.parent.backup(value)
    
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U() - self.illegal_moves_mask * 100000)
    
    def get_move_to_make_for_search(self, ply_count):
        distribution = self.child_number_visits.flatten()
        # Play moves non-deterministically (weighted by distribution)
        # at first, then play the "best" afterwards
        if ply_count > self.nondet_plies:
            return np.unravel_index(np.argmax(distribution), self.child_priors.shape)[0]
        else:
            distribution = softmax(distribution)
            return np.unravel_index(np.random.choice(np.arange(distribution.shape[0]), 1, p=distribution)[0], self.child_priors.shape)[0]