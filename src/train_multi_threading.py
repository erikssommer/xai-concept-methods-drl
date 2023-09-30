import tensorflow as tf
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from utils import config
from rl import mcts_threading
from policy import ActorCriticNet
from env import gogame, govars
import os

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    board_size = config.board_size

    input_shape = (govars.NUM_CHNLS, board_size, board_size)
    output = board_size ** 2 + 1

    move_cap = board_size ** 2 * 5

    model_name = 0

    policy_nn = ActorCriticNet(input_shape, output)

    save_interval = 1

    state_buffer = []
    observation_buffer = []
    value_buffer = []

    epsilon = config.epsilon
    sigma = config.sigma
    c = config.c
    simulations = config.simulations
    episodes = config.episodes


    # Save initial random weights
    policy_nn.model.save(f"../models/board_size_{config.board_size}/net_0")

    for epoch in tqdm(range(0, config.epochs + 1)):
        with Pool(config.nr_of_threads) as p:
            thread_results = p.map(mcts_threading, [
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
        
        if (epoch % save_interval) == 0 or epoch == 1:
            policy_nn.model.save(f"../models/board_size_{config.board_size}/net_{epoch}")
            model_name = epoch
        
        # Update the epsilon and sigma
        epsilon = epsilon * config.epsilon_decay
        sigma = sigma * config.sigma_decay
        