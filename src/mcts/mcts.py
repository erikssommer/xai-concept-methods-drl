import numpy as np
import copy
import random
from .node import Node
from typing import Tuple, List, Any, Union
from game import GoGame, GoVars
import utils

class MCTS:
    def __init__(self, epsilon, sigma, iterations, board_size, c=1.3, policy_nn=None):
        self.iterations = iterations
        self.epsilon = epsilon
        self.sigma = sigma
        self.c = c
        self.policy_nn = policy_nn
        self.board_size = board_size
        self.move_cap = board_size ** 2 * 5

    def __rollout(self, node: Node) -> int:
        """
        Rollout function using epsilon-greedy strategy with default policy
        Not using node structure to save memory
        """
        # Get the current state
        game_state = node.state
        moves = 0

        while not GoGame.game_ended(game_state) and moves < self.move_cap:
            pivot = random.random()

            if pivot < self.epsilon:
                # Random rollout
                action = GoGame.random_action(game_state)

                game_state = GoGame.next_state(game_state, action)

                moves += 1

            else:
                # Rollout using distribution from neural network
                distribution, _ = self.policy_nn.predict(
                    np.array([game_state]))
                
                # Get the valid moves
                valid_moves = GoGame.valid_moves(game_state)

                # Set the invalid moves to 0
                distribution = distribution * valid_moves

                # Softmax the distribution
                distribution = np.exp(distribution) / \
                    np.sum(np.exp(distribution))

                # Get the action
                action = np.argmax(distribution[0])

                # Get the next state
                game_state = GoGame.next_state(game_state, action)

                moves += 1

        # Return the reward of the node given the player using node class even if it is not a terminal state
        return node.winning(self.root.get_player(), game_state)

    def __calculate_ucb1(self, node: Node) -> float:
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0 and node.parent.get_player() == GoVars.BLACK:
            return np.inf
        elif node.visits == 0 and node.parent.get_player() == GoVars.WHITE:
            return -np.inf

        elif node.parent.get_player() == GoVars.BLACK:
            return self.__get_max_value_move(node)
        else:
            return self.__get_min_value_move(node)

    def __get_max_value_move(self, node: Node) -> float:
        """
        Return the max value move for a given node and child
        """
        return self.__q_value(node) + self.__u_value(node)

    def __get_min_value_move(self, node: Node) -> float:
        """
        Return the min value move for a given node and child
        """
        return self.__q_value(node) - self.__u_value(node)

    def __q_value(self, node: Node) -> float:
        """
        Calculate the Q(s,a) value for a given node
        """
        return node.rewards / node.visits

    def __u_value(self, node: Node) -> float:
        """
        Exploration bonus: calculate the U(s,a) value for a given node
        Using upper confidence bound for trees (UCT)
        """
        return self.c * np.sqrt(np.log(node.parent.visits) / (1 + node.visits))

    def __select_best_child(self, node: Node) -> Node:
        """
        Select the best child node using UCB1
        """
        ucb1_scores = [self.__calculate_ucb1(child) for child in node.children]

        best_idx = np.argmax(ucb1_scores) \
            if node.get_player() == GoVars.BLACK \
            else np.argmin(ucb1_scores)

        val = ucb1_scores[best_idx]

        # find all the nodes with the same value
        best_idx = [i for i, j in enumerate(ucb1_scores) if j == val]

        # randomly select one of the best nodes
        best_idx = random.choice(best_idx)

        return node.children[best_idx]

    def __node_expansion(self, node: Node) -> Node:
        # Expand node by adding one of its unexpanded children
        # Get the legal moves from the current state
        node.make_children()

        # Tree policy: return the first child node
        return random.choice(node.children)

    def __simulate(self, node: Node) -> int:
        if random.random() < self.sigma:
            return self.__rollout(node)
        else:
            return self.__critic(node)

    def __critic(self, node: Node):
        game_state = node.state

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
        self.root = Node(state)

    def search(self, starting_player) -> Tuple[Any, Any, List[Union[float, Any]], Any]:
        node: Node = self.root
        node.player = starting_player

        for _ in range(self.iterations):
            leaf_node = self.__tree_search(node)  # Tree policy
            reward = self.__simulate(leaf_node)  # Rollout
            self.__backpropagate(leaf_node, reward)  # Backpropagation

        # Use the edge (from the root) with the highest visit count as the actual move.
        best_move = self.__get_best_move()
        distribution = self.__get_distribution()

        return best_move, best_move.get_player(), copy.deepcopy(best_move.state), distribution

    def reset(self) -> None:
        self.root = None
