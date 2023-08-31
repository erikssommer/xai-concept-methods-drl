import numpy as np
import copy
import random
from mcts.node import Node
from typing import Tuple, List, Any, Union

class MCTS:
    def __init__(self, epsilon, sigma, iterations, c, c_nn=None, dp_nn=None):
        self.iterations = iterations
        self.epsilon = epsilon
        self.sigma = sigma
        self.dp_nn = dp_nn
        self.c_nn = c_nn
        self.c = c

    def __rollout(self, node: Node) -> int:
        """
        Rollout function using epsilon-greedy strategy with default policy
        """

        while not node.state.is_game_over():
            pivot = random.random()

            if pivot < self.epsilon:
                # Random rollout
                node = node.apply_action(random.choice(
                    node.state.get_legal_actions()))
            else:
                # Rollout using default policy
                action = self.dp_nn.rollout_action(node.state)
                try:
                    node = node.apply_action_without_adding_child(action)
                except:
                    self.dp_nn.debug(node.state)

                    node = node.apply_action(random.choice(
                        node.state.get_legal_actions()))
                    raise Exception("Invalid action")

        # Return the reward of the node given the player using node class
        return node.state.get_reward()

    def __calculate_ucb1(self, node: Node) -> float:
        """
        Calculate UCB1 value for a given node and child
        """
        if node.visits == 0 and node.parent.state.get_player() == 1:
            return np.inf
        elif node.visits == 0 and node.parent.state.get_player() == 2:
            return -np.inf

        elif node.parent.state.get_player() == 1:
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
            if node.state.get_player() == 1 \
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
        legal_moves = node.state.get_legal_actions()

        # Expand the node by creating child nodes for each legal move
        for move in legal_moves:
            node.apply_action(move)

        # Tree policy: return the first child node
        return random.choice(node.children)

    def __simulate(self, node: Node) -> int:
        if random.random() < self.sigma:
            return self.__rollout(node)
        else:
            return self.__critic(node)

    def __critic(self, node: Node):
        # TODO: Use the chritic neural network to simulate a playout from the current node
        pass

    def __backpropagate(self, node: Node, reward) -> None:
        # Clear the children of the node generated by the rollout
        node.children = []
        # Backpropagate reward through the tree
        while node is not None:
            node.update(reward)
            node = node.parent

    def __tree_search(self, node: Node) -> Node:
        # Run while the current node is not a leaf node
        while len(node.children) != 0:
            node = self.__select_best_child(node)

        # Test if node is terminal
        if node.state.is_game_over():
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
        return random.choice(best_moves)

    def __get_distribution(self):
        total_visits = sum(child.visits for child in self.root.children)
        dist = [(child.visits / total_visits) for child in self.root.children]

        validity = self.root.state.state.get_validity_of_children()

        distribution = []
        for i in validity:
            if i:
                distribution.append(dist.pop(0))
            else:
                distribution.append(0)

        return distribution

    def set_root(self, state) -> None:
        self.root = Node(state)

    def search(self, starting_player) -> Tuple[Any, Any, List[Union[float, Any]], Any]:
        node: Node = self.root
        node.state.player = starting_player

        for _ in range(self.iterations):
            leaf_node = self.__tree_search(node)  # Tree policy
            reward = self.__simulate(leaf_node)  # Rollout
            self.__backpropagate(leaf_node, reward)  # Backpropagation

        # Use the edge (from the root) with the highest visit count as the actual move.
        best_move = self.__get_best_move()
        distribution = self.__get_distribution()

        return best_move, best_move.state.get_player(), copy.deepcopy(best_move.state.state.game_state), distribution

    def reset(self) -> None:
        self.root = None