import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tqdm import tqdm
from utils import config
from mcts import MCTS
from rbuf import RBUF
from policy import ActorCriticNet
import numpy as np
import gc
from utils import tensorboard_setup, write_to_tensorboard, plot_distribution

import env

def rl():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
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
    go_env = env.GoEnv(size=board_size)

    # Creation replay buffer
    rbuf = RBUF(config.rbuf_size)

    # Create the neural network
    policy_nn = ActorCriticNet(board_size)

    # Create the tensorboard callback
    tensorboard_callback, logdir = tensorboard_setup()

    # Save initial random weights
    policy_nn.save_model(f"../models/training/board_size_{board_size}/net_0.keras")

    # Loop through the number of episodes
    for episode in tqdm(range(config.episodes)):
        # Reset the environment
        go_env.reset()

        # Get the initial state
        game_state = go_env.state()

        # Create the initial tree
        tree = MCTS(game_state, epsilon, sigma, simulations,
                    board_size, move_cap, c, policy_nn)

        # For visualization only
        node = tree.root

        # Play a game until termination
        game_over = False

        while not game_over:
            # Get the player
            curr_player = go_env.turn()
            curr_state = go_env.state()
            best_action_node, distribution = tree.search()

            # Remove array index 3 and 5 from the current state making it an shape of (4, 5, 5)
            curr_state = np.delete(curr_state, [3, 5], axis=0)

            # Add the case to the replay buffer
            rbuf.add_case(curr_player, curr_state, distribution)

            # Apply the action to the environment
            _, _, game_over, _ = go_env.step(
                best_action_node.action)

            if config.render:
                # Render the board
                go_env.render()

            # Update the root node of the mcts tree
            tree.set_root_node(best_action_node)

            # Garbage collection
            gc.collect()

        # Get the winner of the game
        winner = go_env.winning()

        # Set the values of the states
        rbuf.set_values(winner)
        rbuf.clear_lists()
        tree.reset()

        # Train the neural network
        state_buffer, distribution_buffer, value_buffer = zip(
            *rbuf.get(config.batch_size))

        # Train the neural network
        history = policy_nn.fit(
            np.array(state_buffer),
            np.array(distribution_buffer),
            np.array(value_buffer),
            epochs=1,
            callbacks=[tensorboard_callback]
        )

        # Add the metrics to TensorBoard
        write_to_tensorboard(history, episode, logdir)
        
        if episode != 0 and episode % save_interval == 0:
            # Save the neural network model
            policy_nn.save_model(
                f'../models/training/board_size_{board_size}/net_{episode}.keras')

        # Updating sigma and epsilon
        epsilon = epsilon * config.epsilon_decay
        sigma = sigma * config.sigma_decay

        # Garbadge collection
        gc.collect()

    # Save the final neural network model
    policy_nn.save_model(
        f'../models/training/board_size_{board_size}/net_{config.episodes}.keras')
