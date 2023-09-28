import tensorflow as tf
from multiprocessing import Pool
from utils import config
from rl import mcts_threading

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with Pool(config.nr_of_threads) as p:
        results = p.map(mcts_threading, [
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
