"""
Code not in use
"""

import numpy as np

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

    def __init__(self, state, move_indices, move_cap, all_moves, all_moves_inv, cpuct, nondet_plies, prior=0):
        self.state = state
        self.parent = None
        self.move_cap = move_cap
        self.children = {}
        self.expanded = False
        self.player_number = gogame.turn(state)
        self.all_moves = all_moves
        self.all_moves_inv = all_moves_inv
        self.move_indices = move_indices
        self.prior = 0
        self.cpuct = cpuct
        self.nondet_plies = nondet_plies

        self.child_priors = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)
        self.child_win_value = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)
        self.child_number_visits = np.zeros([config.board_size ** 2 + 1], dtype=np.float32)

        valid_moves = gogame.valid_moves(state)
        # Using valid moves as a mask, we can set illegal moves to 0
        self.illegal_moves_mask = np.ones(self.child_priors.shape)
        self.illegal_moves_mask[valid_moves == 0] = 0

    @property
    def number_visits(self):
        # Is root node, all visits go through it
        if self.parent is None:
            return np.sum(self.child_number_visits)
        return self.parent.child_number_visits[self.move_indices]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move_indices] = value

    @property
    def win_value(self):
        return self.parent.child_win_value[self.move_indices]

    @win_value.setter
    def win_value(self, value):
        self.parent.child_win_value[self.move_indices] = value

    def select_leaf(self):
        current = self
        while current.expanded:
            child_move = current.get_best_child()
            # This is a (deferred) leaf, have to create it
            if child_move not in current.children:
                current.add_child(child_move, self.child_priors[child_move])
            current = current.children[child_move]
        return current

    def expand(self, child_priors):
        self.expanded = True
        self.child_priors = child_priors

    def add_child(self, move, prior):
        self.children[move] = Node(gogame.next_state(self.state, move), self.all_moves[move], self.move_cap, self.all_moves, self.all_moves_inv, self.cpuct, self.nondet_plies, prior=prior)
        self.children[move].parent = self

    def update_win_value(self, value):
        if self.parent:
            self.number_visits += 1
            self.win_value += value
            self.parent.update_win_value(value)

    def get_best_child(self):
        if self.player_number == govars.BLACK:
            res = np.argmax(self.child_win_value / (1 + self.child_number_visits) + self.cpuct * self.child_priors * np.sqrt(self.number_visits) * self.illegal_moves_mask)
        else:
            res = np.argmin(self.child_win_value / (1 + self.child_number_visits) - self.cpuct * self.child_priors * np.sqrt(self.number_visits) * self.illegal_moves_mask)

        res = np.unravel_index(res, self.child_priors.shape)
        return res

    def get_move_to_make_for_search(self, ply_count):
        distribution = self.child_number_visits.flatten()
        # Play moves non-deterministically (weighted by distribution)
        # at first, then play the "best" afterwards
        if ply_count > self.nondet_plies:
            return np.unravel_index(np.argmax(distribution), self.child_priors.shape)
        else:
            distribution = softmax(distribution)
            return np.unravel_index(np.random.choice(np.arange(distribution.shape[0]), 1, p=distribution)[0], self.child_priors.shape)