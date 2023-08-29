import graphviz

class Node:
    def __init__(self, state, parent=None, root_node=False, game_state=None):
        if root_node:
            self.state = self.create_root_node_with_state(game_state)
        else:
            self.state = state

        self.parent = parent
        self.children = []
        self.visits = 0
        self.rewards = 0

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

    def create_root_node_with_state(self, state):
        state_manager = None
        state_manager.state.game_state = state
        return state_manager

    def apply_action(self, action):
        """
        Apply an action to the state represented by the node
        """
        # Create a new node representing the next state of the game
        next_node = Node(self.state.apply_action(action), parent=self)

        # Add the new node to the list of children of the current node
        self.children.append(next_node)

        return next_node

    def apply_action_without_adding_child(self, action):
        """
        Apply an action to the state represented by the node without adding the new node to the list of children
        """
        return Node(self.state.apply_action(action), parent=None)

    def visualize_tree(self, graph=None):
        """ 
        Visualize the tree structure of the MCTS tree (for debugging purposes)
        """
        if graph is None:
            graph = graphviz.Digraph()

        graph.node(str(
            id(self)), label=f'Player: {self.state.get_player()}\nVisits: {self.visits}\nRewards: {self.rewards}\nState: {self.state.get_state_flatten()}')

        for child in self.children:
            graph.edge(str(id(self)), str(id(child)))
            child.visualize_tree(graph)
        return graph

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return self.__str__()