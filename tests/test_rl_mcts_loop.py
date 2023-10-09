import unittest
import gc
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

from mcts import MCTS
import env

class TestRlMctsLoop(unittest.TestCase):
    def test(self):
        # Setting the activation of default policy network and critic network
        epsilon = 1
        sigma = 1

        # Set the number of simulations and c constant
        simulations = 50
        c = 1.3
        board_size = 5
        move_cap = board_size ** 2 * 5

        episodes = 10
        visualize_tree = False
        render = False

        # Create the environment
        go_env = env.GoEnv(size=board_size)

        # Loop through the number of episodes
        for _ in tqdm(range(episodes)):
            # Reset the environment
            go_env.reset()

            # Get the initial state
            game_state = go_env.state()

            # Create the initial tree
            tree = MCTS(game_state, epsilon, sigma, simulations,
                        board_size, move_cap, c)

            node = tree.root

            # Play a game until termination
            game_over = False

            while not game_over:
                # Get the player
                best_action_node, distribution = tree.search()

                # Visualize the tree
                if visualize_tree:
                    graph = node.visualize_tree()
                    graph.render(view=True)
                    node = best_action_node

                if render:
                    # Print the distribution
                    print(f"Distribution: {distribution}")
                    # Print valid moves
                    print(f"Valid moves: {go_env.valid_moves()}")
                    # Plot the distribution
                    # utils.plot_distribution(distribution)

                # Apply the action to the environment
                _, _, game_over, _ = go_env.step(
                    best_action_node.action)

                if render:
                    # Render the board
                    go_env.render()

                # Update the root node of the mcts tree
                tree.set_root_node(best_action_node)

                # Garbage collection
                gc.collect()

            # Get the winner of the game
            winner = go_env.winning()

            print(f"Winner: {winner}")

            # Set the values of the states
            tree.reset()

            # Garbadge collection
            gc.collect()

if __name__ == '__main__':
    unittest.main()