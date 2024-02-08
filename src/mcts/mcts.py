import numpy as np
from .node import Node
from typing import Tuple
from env import gogame
import utils
from policy import FastPredictor

class MCTS:
    def __init__(self, game_state, simulations, board_size, move_cap, neural_network: FastPredictor, c=1, komi=0.5, deterministic_moves=8):
        """
        Initialize the Monte Carlo Tree Search

        :param game_state: The initial state of the game
        :param simulations: The number of simulations to run
        :param board_size: The size of the board
        :param move_cap: The maximum number of moves in a game (board_size * board_size * 2 in AlphaGo Zero)
        :param c: The exploration constant
        :param neural_network: The neural network to use for the default policy
        """
        self.root = Node(game_state, player=0)
        self.simulations = simulations
        self.c = c
        self.neural_network = neural_network
        self.board_size = board_size
        self.move_cap = move_cap
        self.komi = komi
        self.deterministic_moves = deterministic_moves

    def __select(self, node: Node) -> Node:
        """
        Select a node to expand

        :param node: The node to select from
        :return: The node to expand
        """
        while node.is_expanded():
            # Otherwise, select the best child
            node = node.best_child(self.c)
        
        return node

    
    def __expand_and_evaluate(self, node: Node):
        """
        Expand a node and evaluate it using the neural network

        :param node: The node to expand
        :return: The value of the node
        """

        # If the node is a terminal node, return the winner
        if gogame.game_ended(node.state):
            if node.player == 1:
                return gogame.winning(node.state, self.komi) * -1
            else:
                return gogame.winning(node.state, self.komi)
        
        # Get valid moves
        valid_moves = gogame.valid_moves(node.state)

        # Creating the state for the neural network
        if node.parent and node.parent.parent:
            prev_turn_state = node.parent.parent.state[0]
        else:
            prev_turn_state = np.zeros((self.board_size, self.board_size))
        
        if node.parent:
            prev_opposing_state = node.parent.state[0]
        else:
            prev_opposing_state = np.zeros((self.board_size, self.board_size))
        
        if node.player == 1:
            state = np.array([node.state[0], prev_turn_state, node.state[1], prev_opposing_state, np.ones((self.board_size, self.board_size))])
        else:
            state = np.array([node.state[0], prev_turn_state, node.state[1], prev_opposing_state, np.zeros((self.board_size, self.board_size))])

        node.predict_state_rep = state
        
        # Use the neural network to get the prior probabilities
        policy, value = self.neural_network.predict(state, valid_moves)

        # Convert valid_moves to a numpy array of integers
        valid_moves_mask = np.array(valid_moves, dtype=int)

        # Make a list of only valid moves
        prior_probabilities = policy[valid_moves_mask == 1]

        node.make_children(prior_probabilities, valid_moves)

        node.expanded = True

        if node.player == 1:
            return value * -1
        else:
            return value
    
    def __backpropagate(self, node: Node, reward: int):
        """
        Backpropagate the reward of a node

        :param node: The node to backpropagate from
        :param reward: The reward to backpropagate
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def __best_action(self, num_moves) -> Tuple[Node, np.ndarray]:
        
        valid_moves_distribution = []

        for child in self.root.children:
            valid_moves_distribution.append(child.n_visit_count)
        
        valid_moves_distribution = utils.normalize(valid_moves_distribution)

        if num_moves > self.deterministic_moves:
            # Choose the best action
            action = np.argmax(valid_moves_distribution)
        else:
            # Choose an action based on the distribution
            action = np.random.choice(len(valid_moves_distribution), p=valid_moves_distribution)

        # Get the node of the best action
        node = self.root.children[action]

        # Get the distribution from the root node including the invalid moves
        distribution = np.zeros(self.board_size ** 2 + 1)
        
        for child in self.root.children:
            distribution[child.action] = child.n_visit_count

        # Normalize the distribution
        distribution = utils.normalize(distribution)

        return node, distribution

    def set_root_node(self, node: Node) -> None:
        self.root = node
        # Remove the reference to the parent node and delete the parent
        self.root.parent = None

    def set_root_node_with_action(self, action) -> None:
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                return
        
        raise Exception("Action not found in children")
    
    def set_root(self, state, player) -> None:
        """Needed in junittests"""
        self.root = None
        del self.root
        self.root = Node(state, player)

    def view_tree(self):
        return self.root.visualize_tree()

    def search(self, num_moves=0) -> Tuple[Node, np.ndarray]:
        """
        Run the Monte Carlo Tree Search

        :return: The best action and the probability of winning
        """
        # Run the simulations
        for _ in range(self.simulations):
            # Select
            node = self.__select(self.root)

            # Expand and evaluate
            value = self.__expand_and_evaluate(node)

            # Backpropagate
            self.__backpropagate(node, value)

        # Return the best action
        return self.__best_action(num_moves)