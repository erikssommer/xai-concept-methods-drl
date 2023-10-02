import tensorflow as tf
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from utils import config
from rl import mcts_threading
from policy import ActorCriticNet
from env import gogame, govars
import os

def setup(board_size):
    # Create the folder containing the models if it doesn't exist
    if not os.path.exists('../models'):
        os.makedirs(f'../models/')
    if not os.path.exists(f'../models/board_size_{board_size}'):
        os.makedirs(f'../models/board_size_{board_size}/')
    else:
        # Delete the model folders
        folders = os.listdir(f'../models/board_size_{board_size}')
        for folder in folders:
            # Test if ends with .keras
            if not folder.endswith('.keras'):
                # Delete the folder even if it's not empty
                os.system(f'rm -rf ../models/board_size_{board_size}/{folder}')
            else:
                # Delete the file
                os.remove(f'../models/board_size_{board_size}/{folder}')

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    board_size = config.board_size

    setup(board_size)

    input_shape = (govars.NUM_CHNLS, board_size, board_size)
    output = board_size ** 2 + 1

    move_cap = board_size ** 2 * 5

    model_name = 0

    policy_nn = ActorCriticNet(input_shape, output)

    save_interval = 2

    state_buffer = []
    observation_buffer = []
    value_buffer = []

    epsilon = config.epsilon
    sigma = config.sigma
    c = config.c
    simulations = config.simulations
    episodes = config.episodes


    # Save initial random weights
    policy_nn.model.save(f"../models/board_size_{config.board_size}/net_0.keras")

    for epoch in tqdm(range(1, config.epochs + 1)):
        with Pool(config.nr_of_threads) as pool:
            thread_results = pool.map(mcts_threading, [
                (
                    thread,
                    model_name,
                    episodes,
                    epsilon,
                    sigma,
                    move_cap,
                    c,
                    simulations,
                    board_size
                )
                for thread in range(config.nr_of_threads)
            ]
            )
        for result in thread_results:
            state_buffer += result[0]
            observation_buffer += result[1]
            value_buffer += result[2]
        
        if epoch > config.epoch_skip:
            state_buffer = state_buffer[-config.rbuf_cap:]
            observation_buffer = observation_buffer[-config.rbuf_cap:]
            value_buffer = value_buffer[-config.rbuf_cap:]

            # Train the neural network
            history = policy_nn.fit(
                np.array(state_buffer),
                np.array(observation_buffer),
                np.array(value_buffer),
                epochs=1
            )

            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay
        
        if (epoch % save_interval) == 0:
            policy_nn.model.save(f"../models/board_size_{config.board_size}/net_{epoch}.keras")
            model_name = epoch
        