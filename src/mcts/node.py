import graphviz
import numpy as np

from game.data import GoGame, GoVars

from utils.read_config import config

class Node:
    def __init__(self, state, parent=None):

        self.player = None

        self.state = state
        self.child_states = None

        self.action = None

        # Node references
        self.parent = parent
        self.children = []

        self.visits = 0
        self.rewards = 0

        self.value = None
        self.first_action = None

        # Level
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def is_game_over(self):
        return GoGame.game_ended(self.state)

    def valid_moves(self):
        return GoGame.valid_moves(self.state)

    def action_size(self):
        return GoGame.action_size(self.state)

    def winning(self, root_player, game_state):
        # Allways in perspective of black
        # Black is 0, white is 1
        win = GoGame.winning(game_state)

        if root_player == GoVars.BLACK and win == 1:
            return 1
        elif root_player == GoVars.WHITE and win == -1:
            return -1
        else:
            return 0
        
    def isleaf(self):
        # Not the same as whether the state is terminal or not
        return (self.child_nodes == None).all()

    def isroot(self):
        return self.parent is None
    
    def get_player(self):
        return self.player

    def make_childnode(self, action, state):
        child_node = Node(state, self)
        
        # Set the player of the child node to the opoposite of the parent node
        child_node.player = 1 - self.player

        # Set the action from the parent to the child node
        child_node.action = action

        # Add the child node to the list of children
        self.children.append(child_node)

        return child_node

    def make_children(self):
        """
        :return: Padded children numpy states
        """
        child_states = GoGame.children(self.state, padded=True)
        actions = np.argwhere(self.valid_moves()).flatten()
        for action in actions:
            self.make_childnode(action, child_states[action])
        self.child_states = child_states

        return child_states
    
    def get_validity_of_children(self):
        dim = config.board_size
        value = np.ones(dim ** 2 + 1)

        """
        for i in range(dim):
            for j in range(dim):
                if self.state[0][i][j] != 0:
                    value[i][j] = 0
        
        for i in range(dim):
            for j in range(dim):
                if self.state[1][i][j] != 0:
                    value[i][j] = 0
        """
        
        # Test if all statates set to 1 are valid
        valid_moves = self.valid_moves()
        
        """
        for i in range(len(valid_moves)):
            if valid_moves[i] == 0:
                value[i] = 0

        # assert if the list of valid moves is the same as the list of value
        assert np.array_equal(valid_moves, value)
            
        return value
        """

        return valid_moves

    def visualize_tree(self, graph=None):
        """ 
        Visualize the tree structure of the MCTS tree (for debugging purposes)
        """
        if graph is None:
            graph = graphviz.Digraph()

        graph.node(str(
            id(self)), label=f'Player: {self.get_player()}\nVisits: {self.visits}\nRewards: {self.rewards}\nState ([black][white]):\n{self.state[0]}\n{self.state[1]}]')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
