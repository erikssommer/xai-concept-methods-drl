import os
from tqdm import tqdm
from utils import config
from mcts import MCTSzero as MCTS
from rbuf import RBUF
from policy import ConvNet, ResNet
import numpy as np
import gc
from utils import tensorboard_setup, write_to_tensorboard

import env

def rl_canonical():

    # Get the config variables
    simulations = config.simulations
    c = config.c
    komi = config.komi
    board_size = config.board_size
    move_cap = board_size ** 2 * 5
    save_interval = config.episodes // config.nr_of_anets
    sample_ratio = config.sample_ratio
    pre_trained = config.pre_trained
    clear_rbuf = config.clear_rbuf
    batch_size = config.batch_size
    rbuf_size = config.rbuf_size
    episodes = config.episodes
    pre_trained_path = config.pre_trained_path

    # Creation replay buffer
    rbuf = RBUF(rbuf_size)

    if pre_trained:
        # Try to get the first file in the pre trained path directory
        try:
            # Get the first file in the directory
            model_name = os.listdir(pre_trained_path)[0]
            model_path = pre_trained_path + model_name
            
            start_episode = int(model_name.split("_")[-1].split(".")[0])
        except:
            # Print error message and exit
            print("Error: Could not find a pre trained model in the specified directory")
            exit()
        
        # Load the neural network
        if config.resnet:
            neural_network = ResNet(board_size, model_path)
        else:
            neural_network = ConvNet(board_size, model_path)
        
    else:
        # Create the neural network
        if config.resnet:
            neural_network = ResNet(board_size)
        else:
            neural_network = ConvNet(board_size)

        # Save initial (random) weights
        neural_network.save_model(f"../models/training/board_size_{board_size}/net_0.keras")
        start_episode = 0

    # Create the tensorboard callback
    tensorboard_callback, logdir = tensorboard_setup()

    # Loop through the number of episodes
    for _ in tqdm(range(start_episode, episodes)):
        # Create the environment
        go_env = env.GoEnv(size=board_size, komi=komi)

        # Reset the environment
        go_env.reset()

        # Get the initial state
        init_state = go_env.canonical_state()

        # Create the initial tree
        tree = MCTS(init_state, simulations, board_size, move_cap, c, komi, neural_network)

        #root = tree.root

        # Play a game until termination
        game_over = False
        # Black always starts
        curr_player = 0

        while not game_over:
            # Get the player
            curr_state = go_env.canonical_state()

            assert curr_state.all() == tree.root.state.all()

            best_action_node, distribution = tree.search()

            #graph = root.visualize_tree()
            #graph.render('./visualization/images/tree', view=True)

            # Add the case to the replay buffer
            if np.random.random() < sample_ratio:
                # Remove array index 3 and 5 from the current state making it an shape of (4, 5, 5)
                curr_state = np.delete(curr_state, [3, 5], axis=0)
                # If current player is 1, change the 2nd array to all 1's
                if curr_player == 1:
                    curr_state[2] = np.ones((board_size, board_size))
                rbuf.add_case(curr_player, curr_state, distribution)

            # Apply the action to the environment
            _, _, game_over, _ = go_env.step(best_action_node.action)

            if config.render:
                # Render the board
                go_env.render()

            # Update the root node of the mcts tree
            tree.set_root_node(best_action_node)

            # Flipp the player
            curr_player = 1 - curr_player

            # Garbage collection
            gc.collect()

        # Get the winner of the game in black's perspective (1 for win and -1 for loss)
        winner = go_env.winner()

        assert winner != 0

        if config.render:
            print(f"Winner: {winner}")

        # Set the values of the states
        rbuf.set_values(winner)

        # Train the neural network
        state_buffer, distribution_buffer, value_buffer = zip(
            *rbuf.get(batch_size))

        # Train the neural network
        history = neural_network.fit(
            np.array(state_buffer),
            np.array(distribution_buffer),
            np.array(value_buffer),
            epochs=1,
            callbacks=[tensorboard_callback]
        )

        # Add the metrics to TensorBoard
        write_to_tensorboard(history, start_episode, logdir)
        
        if start_episode != 0 and start_episode % save_interval == 0:
            # Save the neural network model
            neural_network.save_model(
                f'../models/training/board_size_{board_size}/net_{start_episode}.keras')

        # For every 100 episode, delete the rbuf
        if clear_rbuf:
            if start_episode % 100 == 0:
                del rbuf
                rbuf = RBUF(rbuf_size)

        # Delete references and garbadge collection        
        del tree.root
        del tree
        del go_env
        gc.collect()

        start_episode += 1

    # Save the final neural network model
    neural_network.save_model(
        f'../models/training/board_size_{board_size}/net_{episodes}.keras')