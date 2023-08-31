import graphviz

class Node:
    def __init__(self, state, parent=None):
        
        self.state = state
        self.parent = parent
        self.child_states = None

        self.visits = 0
        self.rewards = 0

        # Level
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

    def update(self, reward):
        self.visits += 1
        self.rewards += reward

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