from tqdm import tqdm
from utils.read_config import config
from mcts.mcts import MCTS
from rbuf.rbuf import RBUF
from models.policy_nn import ActorCriticNet
import gym
import numpy as np
import gc


class RL:

    def learn(self):

        # Setting the activation of default policy network and critic network
        epsilon = config.epsilon
        sigma = config.sigma

        # Set the number of simulations and c constant
        simulations = config.simulations
        c = config.c

        save_interval = config.episodes // config.nr_of_anets

        # Create the environment
        go_env: gym.Env = gym.make('gym_go:go-v0', size=config.board_size)

        # Creation replay buffer
        rbuf = RBUF(config.rbuf_size)

        print(go_env.observation_space.shape)

        # Create the neural network
        policy_nn = ActorCriticNet(go_env.observation_space.shape, config.move_cap)

        # Loop through the number of episodes
        for episode in tqdm(range(config.episodes)):

            # Reset the environment
            go_env.reset()

            # Get the initial state
            game_state = go_env.canonical_state()

            # Create the initial tree
            tree = MCTS(epsilon, sigma, simulations, c)

            # Set the root node of the tree
            tree.set_root(game_state)

            # For visualization only
            node = tree.root

            # Play a game until termination
            terminated = False

            while not terminated:
                # Get the player
                curr_player = go_env.turn()

                best_action_node, player, game_state, distribution = tree.search(
                    curr_player)
                
                #print(distribution)
                
                # Visualize the tree
                if config.visualize_tree:
                    graph = node.visualize_tree()
                    graph.render('./visualization/tree', view=True)
                    node = best_action_node

                # Add to rbuf (replay buffer)
                rbuf.add_case((curr_player, game_state, distribution))

                # Apply the action to the environment
                observation, reward, terminated, info = go_env.step(
                    best_action_node.action)

                # Render the board
                # go_env.render()

                # Update the root node of the mcts tree
                tree.root = best_action_node

                # Garbage collection
                gc.collect()

            # Get the winner of the game
            winner = go_env.winning()

            # Set the values of the states
            rbuf.set_values(winner)

            tree.reset()

            # Updating sigma and epsilon
            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay

            # Train the neural network
            state_buffer, distribution_buffer, value_buffer = zip(*rbuf.get(config.batch_size))

            # Train the neural network
            policy_nn.fit(np.array(state_buffer), np.array(distribution_buffer), np.array(value_buffer), epochs=1)

            # Save the neural network model
            if episode % save_interval == 0 and episode != 0:
                # Save the neural network model
                policy_nn.save_weights(f'../models/rl_policy_nn_{episode}.h5')

            # Garbadge collection
            gc.collect()
        
        # Save the final neural network model
        policy_nn.save_weights(f'../models/rl_policy_nn_{config.episodes}.h5')
            

        print("Finished training")
