import numpy as np
from .node import Node
from typing import Tuple
from env import gogame
import utils
from policy import FastPredictor


class MCTS:
    def __init__(self,
                 game_state: np.ndarray,
                 simulations: int,
                 board_size: int,
                 move_cap: int,
                 neural_network: FastPredictor,
                 c: float = 1,
                 komi: float = 0.5,
                 non_det_moves: int = 8):
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
        self.non_det_moves = non_det_moves

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

    def __expand_and_evaluate(self, node: Node) -> float:
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

        model_state = node.model_state_format(self.board_size)

        # Use the neural network to get the prior probabilities
        policy, value = self.neural_network.predict(model_state, valid_moves)

        # Convert valid_moves to a numpy array of integers
        valid_moves_mask = np.array(valid_moves, dtype=int)

        # Make a list of only valid moves
        prior_probabilities = policy[valid_moves_mask == 1]

        # Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node s0
        # Specifically P(s, a) = (1 − ε)pa + εηa, where η ∼ Dir(0.03) and ε = 0.25; as in the AlphaGo Zero paper
        # Explination from paper: "this noise ensures that all moves may be tried, but the search may still overrule bad moves."
        cnoise = 0.25
        noise = np.random.dirichlet([0.03] * len(prior_probabilities))
        prior_probabilities = (1 - cnoise) * \
            prior_probabilities + cnoise * noise
        prior_probabilities /= prior_probabilities.sum()

        node.make_children(prior_probabilities, valid_moves)

        node.expanded = True

        if node.player == 1:
            return value * -1
        else:
            return value

    def __backpropagate(self, node: Node, reward: int) -> None:
        """
        Backpropagate the reward of a node

        :param node: The node to backpropagate from
        :param reward: The reward to backpropagate
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def __best_action(self, num_moves: int) -> Tuple[Node, np.ndarray]:

        valid_moves_distribution = []

        for child in self.root.children:
            valid_moves_distribution.append(child.n_visit_count)

        valid_moves_distribution = utils.normalize(valid_moves_distribution)

        if num_moves > self.non_det_moves:
            # Choose the best action
            action = np.argmax(valid_moves_distribution)
        else:
            # Choose an action based on the distribution
            action = np.random.choice(
                len(valid_moves_distribution), p=valid_moves_distribution)

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

    def set_root_node_with_action(self, action: int) -> None:
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                return

        raise Exception("Action not found in children")

    def set_root(self, state: np.ndarray, player: int) -> None:
        """Needed in junittests"""
        self.root = None
        del self.root
        self.root = Node(state, player)

    def reset_root(self):
        self.root.reset()

    def view_tree(self):
        return self.root.visualize_tree()

    def search(self, num_moves: int=0) -> Tuple[Node, np.ndarray]:
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
