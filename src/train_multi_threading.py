import tensorflow as tf
import numpy as np
from multiprocessing import Pool
from utils import config
from rl import mcts_threading
from policy import ActorCriticNet
from env import gogame, govars

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    
    board_size = config.board_size

    input_shape = (govars.NUM_CHNLS, board_size, board_size)
    output = board_size ** 2 + 1

    move_cap = board_size ** 2 * 5

    policy_nn = ActorCriticNet(input_shape, output)

    save_interval = config.episodes // config.nr_of_anets

    state_buffer = []
    observation_buffer = []
    value_buffer = []

    # Save initial random weights
    policy_nn.model.save(f"../models/board_size_{config.board_size}/net_0")

    for epoch in range(1, config.epochs + 1):
        with Pool(config.nr_of_threads) as p:
            thread_results = p.map(mcts_threading, [
                (
                    thread,
                    config.episodes,
                    config.epsilon,
                    config.sigma,
                    config.c,
                    config.simulations,
                    config.board_size
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

            history = policy_nn.fit(
                np.array(observation_buffer),
                np.array(state_buffer),
                np.array(value_buffer),
                epochs=1
            )
        
        if (epoch % save_interval) == 0 or epoch == 1:
            policy_nn.model.save(f"../models/board_size_{config.board_size}/net_{epoch}")
        