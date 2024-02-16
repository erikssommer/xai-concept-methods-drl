from __future__ import annotations
import graphviz
import numpy as np

from env import gogame


class Node:
    def __init__(self,
                 state: np.ndarray,
                 player: int,
                 parent: Node = None,
                 prior_probability: int = 0,
                 time_step: int = 0):

        # Game state
        self.player = player
        self.state = state
        self.action = None

        # Node references
        self.parent = parent
        self.children: list[Node] = []

        # MCTS values
        self.n_visit_count = 0
        self.w_total_action_value = 0
        self.q_mean_action_value = 0
        self.p_prior_probability = prior_probability

        # Node metadata
        self.expanded = False
        self.time_step = time_step
        self.predict_state_rep: np.ndarray = None
        self.optiaml_rollout = None

    def is_expanded(self) -> bool:
        return self.expanded

    def best_child(self, c: float) -> Node:
        """
        Select the best child node using PUCT algorithm
        """
        if self.player == 0:
            index = np.argmax([child.q_value() + child.u_value(c)
                              for child in self.children])
        else:
            index = np.argmin([child.q_value() - child.u_value(c)
                              for child in self.children])

        # Return the best child
        return self.children[index]

    def update(self, reward: float) -> None:
        self.n_visit_count += 1
        self.w_total_action_value += reward
        self.q_mean_action_value = self.w_total_action_value / self.n_visit_count

    def q_value(self) -> float:
        """
        Calculate the Q(s,a) value for a given node
        """
        return self.q_mean_action_value

    def u_value(self, c: float) -> float:
        """
        Exploration bonus: calculate the U(s,a) value for a given node
        Using upper confidence bound for trees (UCT)
        """
        return c * self.p_prior_probability * np.sqrt(self.parent.n_visit_count) / (1 + self.n_visit_count)

    def make_children(self, prior_probabilities: list, valid_moves: np.ndarray) -> None:
        """
        :return: Padded children numpy states
        """
        child_states = gogame.children(self.state, canonical=True)

        actions = np.argwhere(valid_moves).flatten()

        child_player = 1 - self.player
        child_timestep = self.time_step + 1

        # Using enumerate to get the index of the action
        for i, action in enumerate(actions):
            child_node = Node(state=child_states[action],
                              player=child_player,
                              parent=self,
                              prior_probability=prior_probabilities[i],
                              time_step=child_timestep)

            # Set the action from the parent to the child node
            child_node.action = action

            # Add the child node to the list of children
            self.children.append(child_node)

    def model_state_format(self, board_size: int) -> np.ndarray:
        # Creating the state for the neural network
        if self.parent and self.parent.parent:
            prev_turn_state = self.parent.parent.state[0]
        else:
            prev_turn_state = np.zeros((board_size, board_size))

        if self.parent:
            prev_opposing_state = self.parent.state[0]
        else:
            prev_opposing_state = np.zeros((board_size, board_size))

        if self.player == 1:
            state = np.array([self.state[0], prev_turn_state, self.state[1],
                             prev_opposing_state, np.ones((board_size, board_size))])
        else:
            state = np.array([self.state[0], prev_turn_state, self.state[1],
                             prev_opposing_state, np.zeros((board_size, board_size))])

        self.predict_state_rep = state

        return state

    def reset(self) -> None:
        self.n_visit_count = 0
        self.w_total_action_value = 0
        self.q_mean_action_value = 0
        self.p_prior_probability = 0
        self.expanded = False
        self.action = None
        self.parent = None
        self.children = []

    def visualize_tree(self, graph: graphviz.Digraph = None) -> graphviz.Digraph:
        if graph is None:
            graph = graphviz.Digraph()

        graph.node(str(
            id(self)), label=f'Player: {self.player}\n' +
            f'Timestep: {self.time_step}\n' +
            f'Visits: {self.n_visit_count}\n' +
            f'Rewards: {self.q_mean_action_value}\n' +
            f'State ([black][white]):\n{self.state[0]}\n{self.state[1]}')

        # Color the node red if not optimal and green if optimal
        if self.optiaml_rollout is not None:
            if self.optiaml_rollout:
                graph.node(str(id(self)), style='filled', fillcolor='green')
            else:
                graph.node(str(id(self)), style='filled', fillcolor='red')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()
