from tqdm import tqdm
from utils.read_config import config
from mcts.mcts import MCTS
from rbuf.rbuf import RBUF
from models.policy_nn import ActorCriticNet
import gym


class RL:

    def learn(self):

        # Setting the activation of default policy network and critic network
        epsilon = config.epsilon
        sigma = config.sigma

        # Set the number of simulations and c constant
        simulations = config.simulations
        c = config.c

        # Creation replay buffer
        rbuf = RBUF(config.rbuf_size)

        # Create the neural network
        policy_nn = ActorCriticNet(config.board_size)

        # Loop through the number of episodes
        for episode in tqdm(range(config.episodes)):

            # Create the environment
            go_env: gym.Env = gym.make('gym_go:go-v0', size=config.board_size)

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
                
                # Visualize the tree
                if config.visualize_tree:
                    graph = node.visualize_tree()
                    graph.render('./visualization/tree', view=True)
                    node = best_action_node

                # Add to rbuf (replay buffer)
                rbuf.add_case((player, game_state, distribution))

                # Apply the action to the environment
                observation, reward, terminated, info = go_env.step(
                    best_action_node.action)

                # Render the board
                # go_env.render()

                # Update the root node of the mcts tree
                tree.root = best_action_node

            tree.reset()

            # Updating sigma and epsilon
            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay

            # Train the neural network
            batch = rbuf.get(config.batch_size)

            # Train the neural network
            policy_nn.fit(batch)

            # Save the neural network model
            if episode % config.save_interval == 0:
                # Save the neural network model
                policy_nn.save_weights(f'../models/rl_policy_nn_{episode}.h5')
        
        # Save the final neural network model
        policy_nn.save_weights(f'../models/rl_policy_nn_{config.episodes}.h5')
            

        print("Finished training")
