import numpy as np

from mcts.node_uct import Node
from env import gogame, govars
from utils import config

class MonteCarlo:

    def __init__(self, root_node: Node, move_cap, dims, cpuct, komi=0.5, net=None):
        self.root_node = root_node
        self.move_cap = move_cap
        self.net = net
        self.cpuct = cpuct
        self.dims = dims
        self.komi = komi

    def distribution(self):
        return self.root_node.child_number_visits

    def move_root(self, node, cut_parent=True):
        self.root_node = node
        if cut_parent:
            self.root_node.parent = None
    
    def make_choice(self, current_ply):
        """
        Choose a move to play based on the current distribution
        """
        best_move = self.root_node.get_move_to_make_for_search(current_ply)
        if best_move not in self.root_node.children:
            self.root_node.add_child(best_move)
        return self.root_node.children[best_move]

    def simulate(self):
        leaf = self.root_node.select_leaf()
        # If the leaf is a terminal node, just return the actual result
        if gogame.game_ended(leaf.state):
            reward = gogame.winning(leaf.state)
            leaf.backup(reward if self.root_node.player_number == govars.BLACK else -reward)
            return

        # The probability of going to each child-node, in addition to a value_estimate of the current one
        child_priors, value_estimate = self.net.predict(leaf.state)

         # Get valid moves
        valid_moves = gogame.valid_moves(leaf.state)


        # Convert valid_moves to a numpy array of integers
        valid_moves_mask = np.array(valid_moves, dtype=int)

        # Set the invalid moves to 0
        child_priors[valid_moves_mask == 0] = 0

        # Normalize the child_priors
        child_priors = child_priors / np.sum(child_priors)

        leaf.expand(child_priors)
        leaf.backup(value_estimate)