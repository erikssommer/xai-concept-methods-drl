import graphviz
import numpy as np

from game.data import GoGame


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

    def winning(self):
        return GoGame.winning(self.state)

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
        child_states = GoGame.children(self.state, canonical=True, padded=True)
        actions = np.argwhere(self.valid_moves()).flatten()
        for action in actions:
            self.make_childnode(action, child_states[action])
        self.child_states = child_states

        return child_states

    def apply_random_move(self):
        """
        Apply an action to the state represented by the node
        """
        child_states = GoGame.children(self.state, canonical=True, padded=True)
        actions = np.argwhere(self.valid_moves()).flatten()

        # Make a childnode for only one random action
        action = np.random.choice(actions)

        node = self.make_childnode(action, child_states[action])

        return node

    def visualize_tree(self, graph=None):
        """ 
        Visualize the tree structure of the MCTS tree (for debugging purposes)
        """
        if graph is None:
            graph = graphviz.Digraph()

        graph.node(str(
            id(self)), label=f'Player: {self.get_player()}\nVisits: {self.visits}\nRewards: {self.rewards}\nState: {self.state}')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
