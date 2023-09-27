#import graphviz
import numpy as np

from env import gogame
from env import govars

class Node:
    def __init__(self, state: np.ndarray, parent=None):

        # Game state
        self.player = int(state[2][0][0])
        self.state = state
        self.action = None

        # Node references
        self.parent = parent
        self.children = []

        # MCTS values
        self.visits = 0
        self.rewards = 0

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def is_game_over(self):
        return gogame.game_ended(self.state)
    
    def q_value(self) -> float:
        """
        Calculate the Q(s,a) value for a given node
        """
        return self.rewards / self.visits
    
    def u_value(self) -> float:
        """
        Exploration bonus: calculate the U(s,a) value for a given node
        Using upper confidence bound for trees (UCT)
        """
        return np.sqrt(np.log(self.parent.visits) / (1 + self.visits))

    def winning(self, root_player, game_state):
        # Allways in perspective of black
        # Black is 0, white is 1
        # 0 is in game or draw, 1 is black win, -1 is black loss
        win = gogame.winning(game_state)

        if root_player == govars.BLACK and win == 1:
            return 1
        elif root_player == govars.WHITE and win == -1:
            return -1
        elif root_player == govars.BLACK and win == -1:
            return -1
        elif root_player == govars.WHITE and win == 1:
            return 1
        else:
            return 0

    def make_childnode(self, action, state):
        child_node = Node(state, self)

        # Set the action from the parent to the child node
        child_node.action = action

        # Add the child node to the list of children
        self.children.append(child_node)

        return child_node

    def make_children(self):
        """
        :return: Padded children numpy states
        """
        child_states = gogame.children(self.state)
        valid_moves = gogame.valid_moves(self.state)

        actions = np.argwhere(valid_moves).flatten()

        for action in actions:
            self.make_childnode(action, child_states[action])

    """ Visualize the tree structure of the MCTS tree (for debugging purposes)
    def visualize_tree(self, graph=None):
        if graph is None:
            graph = graphviz.Digraph()

        graph.node(str(
            id(self)), label=f'Player: {self.player}\nVisits: {self.visits}\nRewards: {self.rewards}\nState ([black][white]):\n{self.state[0]}\n{self.state[1]}]')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph
    """

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
