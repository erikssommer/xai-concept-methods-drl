from tqdm import tqdm
from utils import config
from mcts import MCTS
from rbuf import RBUF
from models import ActorCriticNet
import gym
import numpy as np
import gc
import logging

# Set the logging level
gym.logger.set_level(40)

logger = logging.getLogger(__name__)


class RL:

    def learn(self):

        logger.info("RL training loop started")

        # Setting the activation of default policy network and critic network
        epsilon = config.epsilon
        sigma = config.sigma

        # Set the number of simulations and c constant
        simulations = config.simulations
        c = config.c

        board_size = config.board_size
        move_cap = board_size ** 2 * 5

        save_interval = config.episodes // config.nr_of_anets

        # Create the environment
        go_env: gym.Env = gym.make('gym_go:go-v0', size=board_size)

        # Creation replay buffer
        rbuf = RBUF(config.rbuf_size)

        # Create the neural network (+1 is for the pass action)
        policy_nn = ActorCriticNet(
            go_env.observation_space.shape, config.board_size ** 2 + 1)

        # Loop through the number of episodes
        for episode in tqdm(range(config.episodes)):

            # Reset the environment
            go_env.reset()

            # Get the initial state
            game_state = go_env.canonical_state()

            # Create the initial tree
            tree = MCTS(game_state, epsilon, sigma, simulations, board_size, move_cap, c, policy_nn)

            # For visualization only
            node = tree.root

            # Play a game until termination
            terminated = False

            while not terminated:
                # Get the player
                curr_player = go_env.turn()

                best_action_node, game_state, distribution = tree.search()

                # Visualize the tree
                if config.visualize_tree:
                    graph = node.visualize_tree()
                    graph.render('../../log/visualization/tree', view=True)
                    node = best_action_node

                # Add to rbuf (replay buffer)
                rbuf.add_case((curr_player, game_state, distribution))

                # Apply the action to the environment
                observation, reward, terminated, info = go_env.step(
                    best_action_node.action)

                # Render the board
                if config.render:
                    # Print the distribution
                    print(f"Distribution: {distribution}")
                    # Print valid moves
                    print(f"Valid moves: {go_env.valid_moves()}")
                    go_env.render()

                # Update the root node of the mcts tree
                tree.root = best_action_node
                #tree.set_root(game_state)

                # Garbage collection
                gc.collect()

            # Get the winner of the game
            winner = go_env.winning()

            print(f"Winner: {winner}")

            # Set the values of the states
            rbuf.set_values(winner)

            tree.reset()

            # Train the neural network
            state_buffer, distribution_buffer, value_buffer = zip(
                *rbuf.get(config.batch_size))

            # Train the neural network
            policy_nn.fit(np.array(state_buffer), np.array(
                distribution_buffer), np.array(value_buffer), epochs=10)

            # Save the neural network model
            if episode % save_interval == 0 and episode != 0:
                # Save the neural network model
                policy_nn.model.save(f'../models/actor_critic_net_{episode}')
            
            # Updating sigma and epsilon
            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay

            # Garbadge collection
            gc.collect()

        # Save the final neural network model
        policy_nn.model.save(f'../models/actor_critic_net_{config.episodes}')

        logger.info("RL training loop ended")
