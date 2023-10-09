import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from utils import config
from mcts import mcts_loop
from policy import ActorCriticNet
from utils import tensorboard_setup, write_to_tensorboard

def rl_multiprocessing():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    board_size = config.board_size

    move_cap = board_size ** 2 * 5

    model_name = 0

    policy_nn = ActorCriticNet(board_size)

    save_interval = 2

    state_buffer = []
    observation_buffer = []
    value_buffer = []

    epsilon = config.epsilon
    sigma = config.sigma
    c = config.c
    simulations = config.simulations
    episodes = config.episodes

    tensorboard_callback, logdir = tensorboard_setup()

    # Save initial random weights
    policy_nn.save_model(f"../models/training/board_size_{config.board_size}/net_0.keras")

    for epoch in tqdm(range(1, config.epochs + 1)):
        with Pool(config.nr_of_threads) as pool:
            thread_results = pool.map(mcts_loop, [
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
                epochs=1,
                callbacks=[tensorboard_callback]
            )

            # Add the metrics to TensorBoard
            write_to_tensorboard(history, epoch, logdir)

            epsilon = epsilon * config.epsilon_decay
            sigma = sigma * config.sigma_decay
        
        if (epoch % save_interval) == 0:
            policy_nn.save_model(f"../models/training/board_size_{config.board_size}/net_{epoch}.keras")
            model_name = epoch
    
    # Save the final neural network model
    policy_nn.save_model(f"../models/training/board_size_{config.board_size}/net_{config.epochs}.keras")