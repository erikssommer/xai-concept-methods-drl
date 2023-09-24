import numpy as np
import random
from .node import Node
from typing import Tuple, List, Any
from env import govars
from env import gogame
import utils

class MCTS:
    def __init__(self, game_state, epsilon, sigma, simulations, board_size, move_cap, c=1.3, policy_nn=None):
        self.root = Node(game_state)
        self.simulations = simulations
        self.epsilon = epsilon
        self.sigma = sigma
        self.c = c
        self.policy_nn = policy_nn
        self.board_size = board_size
        self.move_cap = move_cap

    def __rollout(self, game_state: np.ndarray) -> int:
        """
        Rollout function using epsilon-greedy strategy with default policy
        Not using node structure to save memory
        """
        moves = 0

        while not gogame.game_ended(game_state) and moves < self.move_cap:
            pivot = random.random()

            if pivot < self.epsilon:
                # Random rollout
                action = gogame.random_action(game_state)

                game_state = gogame.next_state(game_state, action)

                moves += 1

            else:
                # Rollout using distribution from neural network
                distribution, _ = self.policy_nn.predict(
                    np.array([game_state]))
                
                # Get the valid moves
                valid_moves = gogame.valid_moves(game_state)

                # Set the invalid moves to 0
                distribution = distribution * valid_moves

                # Softmax the distribution
                distribution = np.exp(distribution) / \
                    np.sum(np.exp(distribution))

                # Get the action
                action = np.argmax(distribution[0])

                # Get the next state
                game_state = gogame.next_state(game_state, action)

                moves += 1

        # Return the reward of the node given the player using node class even if it is not a terminal state
        return self.root.winning(self.root.player, game_state)

    def __calculate_ucb1(self, node: Node) -> float:
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0 and node.parent.player == govars.BLACK:
            return np.inf
        elif node.visits == 0 and node.parent.player == govars.WHITE:
            return -np.inf

        elif node.parent.player == govars.BLACK:
            return self.__get_max_value_move(node)
        else:
            return self.__get_min_value_move(node)

    def __get_max_value_move(self, node: Node) -> float:
        """
        Return the max value move for a given node and child
        """
        return node.q_value() + self.c * node.u_value()

    def __get_min_value_move(self, node: Node) -> float:
        """
        Return the min value move for a given node and child
        """
        return node.q_value() - self.c * node.u_value()

    def __select_best_child(self, node: Node) -> Node:
        """
        Select the best child node using UCB1
        """
        ucb1_scores = [self.__calculate_ucb1(child) for child in node.children]

        if node.player == govars.BLACK:
            best_idx = np.argmax(ucb1_scores)
        else:
            best_idx = np.argmin(ucb1_scores)

        value = ucb1_scores[best_idx]

        # Find all the nodes with the same value
        best_idx = [i for i, j in enumerate(ucb1_scores) if j == value]

        # Randomly select one of the best nodes
        best_idx = random.choice(best_idx)

        return node.children[best_idx]

    def __node_expansion(self, node: Node) -> Node:
        # Expand node by adding one of its unexpanded children
        # Get the legal moves from the current state
        node.make_children()

        # Tree policy: return the first child node
        return random.choice(node.children)

    def __simulate(self, game_state: np.ndarray) -> int:
        if random.random() < self.sigma:
            return self.__rollout(game_state)
        else:
            return self.__critic(game_state)

    def __critic(self, game_state: np.ndarray) -> float:

        # Get the value from the policy network
        _, value = self.policy_nn.predict(np.array([game_state]))

        # Get the value from the tensor
        value = 1 - value.numpy()[0][0]

        return value

    def __backpropagate(self, node: Node, reward) -> None:
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent
            #reward = 1 - reward

    def __tree_search(self, node: Node) -> Node:
        # Run while the current node is not a leaf node
        while len(node.children) != 0:
            node = self.__select_best_child(node)

        # Test if node is terminal
        if node.is_game_over():
            return node

        # Test if node has been visited before or if it is the root node
        if node.visits == 1 or node == self.root:
            # For each available action from the current state, create a child node and add it to the tree
            return self.__node_expansion(node)

        # Return the node to be simulated (rollout)
        return node

    def __get_best_move(self) -> Node:
        max_visits = max(self.root.children, key=lambda c: c.visits).visits
        best_moves = [
            child for child in self.root.children if child.visits == max_visits]
        # Add some randomness and not always choose the same move eagerly
        # Could use argmax
        return random.choice(best_moves)

    def __get_distribution(self):
        # Get the distribution from the root node
        distribution = np.zeros(self.board_size ** 2 + 1)
        
        for child in self.root.children:
            distribution[child.action] = child.visits

        # Softmax the distribution
        distribution = utils.normalize(distribution)

        return distribution

    def set_root(self, state) -> None:
        del self.root
        self.root = Node(state)

    def search(self) -> Tuple[Node, Any, List[float]]:
        for _ in range(self.simulations):
            leaf_node = self.__tree_search(self.root)  # Tree policy
            reward = self.__simulate(leaf_node.state)  # Rollout
            self.__backpropagate(leaf_node, reward)  # Backpropagation

        # Use the edge (from the root) with the highest visit count as the actual move.
        best_move = self.__get_best_move()
        distribution = self.__get_distribution()

        return best_move, best_move.state, distribution

    def reset(self) -> None:
        del self.root
        self.root = None
