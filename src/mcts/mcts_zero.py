import numpy as np
from .node_zero import Node
from typing import Tuple, List
from env import govars
from env import gogame
import utils
from policy import ActorCriticNet
import random

class MCTSzero:
    def __init__(self, game_state, simulations, board_size, move_cap, c=1.3, komi=0.5, policy_nn: ActorCriticNet=None):
        """
        Initialize the Monte Carlo Tree Search

        :param game_state: The initial state of the game
        :param simulations: The number of simulations to run
        :param board_size: The size of the board
        :param move_cap: The maximum number of moves in a game (board_size * board_size * 2 in AlphaGo Zero)
        :param c: The exploration constant
        :param policy_nn: The neural network to use for the default policy
        """
        self.root = Node(game_state)
        self.simulations = simulations
        self.c = c
        self.policy_nn = policy_nn
        self.board_size = board_size
        self.move_cap = move_cap
        self.komi = komi

    def __select(self, node: Node) -> Node:
        """
        Select a node to expand

        :param node: The node to select from
        :return: The node to expand
        """
        # If the node is not fully expanded, return it
        if not node.is_expanded():
            return node

        # Otherwise, select the best child
        return self.__select(node.best_child(self.c))

    
    def __expand_and_evaluate(self, node: Node):
        """
        Expand a node and evaluate it using the neural network

        :param node: The node to expand
        :return: The value of the node
        """

        # If the node is a terminal node, return the winner
        if node.is_game_over():
            if gogame.turn(node.state) == govars.BLACK:
                return gogame.winning(node.state, self.komi)
            else:
                return gogame.winning(node.state, self.komi) * -1
        
        # Use the neural network to get the prior probabilities
        policy, value = self.policy_nn.predict(node.state)

        # Make a list of only valid moves
        prior_probabilities = []
        for i in range(len(policy)):
            if policy[i] != 0:
                prior_probabilities.append(i)
        
        node.make_children(prior_probabilities)

        node.expanded = True

        return value

    def __evaluate(self, node: Node) -> int:
        """
        Evaluate a node using the value head of the neural network

        :param node: The node to evaluate
        :return: The reward of the node
        """

        # Get the value from the policy network
        value = self.policy_nn.predict(node.state, value_only=True)

        return value
    
    def __backpropagate(self, node: Node, reward: int):
        """
        Backpropagate the reward of a node

        :param node: The node to backpropagate from
        :param reward: The reward to backpropagate
        """
        # Update the node
        node.update(reward)

        # If the node has a parent, backpropagate from it
        if node.parent:
            self.__backpropagate(node.parent, reward)

    def __best_action(self) -> Tuple[np.ndarray, float]:
        max_visits = max(child.n_visit_count for child in self.root.children)
        best_moves = [
            child for child in self.root.children if child.n_visit_count == max_visits]
        
        # Add some randomness and not always choose the same move eagerly
        node = random.choice(best_moves)

        # Get the distribution from the root node
        distribution = np.zeros(self.board_size ** 2 + 1)
        
        for child in self.root.children:
            distribution[child.action] = child.n_visit_count

        # Softmax the distribution
        distribution = utils.normalize(distribution)

        return node, distribution

    def set_root_node(self, node: Node) -> None:
        self.root = node
        # Remove the reference to the parent node and delete the parent
        self.root.parent = None

    def search(self) -> Tuple[np.ndarray, float]:
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
        return self.__best_action()