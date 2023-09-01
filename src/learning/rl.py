from tqdm import tqdm
from utils.read_config import config
from mcts.mcts import MCTS
import gym


class RL:

    def learn(self):

        # Setting the activation of default policy network and critic network
        epsilon = config.epsilon
        sigma = config.sigma

        # Set the number of simulations and c constant
        simulations = config.simulations
        c = config.c

        # Loop through the number of episodes
        for _ in tqdm(range(config.episodes)):

            # Create the environment
            go_env: gym.Env = gym.make('gym_go:go-v0', size=config.board_size)

            # Reset the environment
            go_env.reset()

            # Get the initial state
            game_state = go_env.canonical_state()

            # Get the player
            curr_player = go_env.turn()

            # Create the initial tree
            tree = MCTS(epsilon, sigma, simulations, c)

            # Set the root node of the tree
            tree.set_root(game_state)

            # Play a game until termination
            terminated = False

            while not terminated:
                best_action_node, player, game_state, distribution = tree.search(
                    curr_player)
                
                # Visualize the tree
                if config.visualize_tree:
                    graph = best_action_node.visualize_tree()
                    graph.render('./visualization/tree', view=True)

                # TODO add to rbuf (replay buffer)

                # Apply the action to the environment
                observation, reward, terminated, info = go_env.step(
                    best_action_node.action)

                # Render the board
                go_env.render()

                # Update the root node of the mcts tree
                tree.root = best_action_node

            tree.reset()

            # Updating sigma and epsilon
            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay

            # TODO train the networks

        print("Finished training")
