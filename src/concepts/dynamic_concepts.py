import numpy as np
from mcts import MCTSzero as MCTS
from policy import ConvNet, FastPredictor, LiteModel

"""
Dynamic concepts are concepts that are not fixed, but rather change over time in response to the environment.
"""


SIMULATION_COUNT = 500
BOARD_SIZE = 5
KOMI = 0.5
MOVE_CAP = BOARD_SIZE ** 2 * 4
C = 1
DETERMINISTIC_MOVES = 0
BEST_MODEL_PATH = f"../models/best_models/board_size_{BOARD_SIZE}/net_10000.keras"

neural_network = ConvNet(BOARD_SIZE, load_path=BEST_MODEL_PATH)

predictor = FastPredictor(LiteModel.from_keras_model(neural_network.model))

def capture_stones_sequence():
    init_state = np.array([])

    # Initialize the MCTS
    mcts = MCTS(init_state, SIMULATION_COUNT, BOARD_SIZE, MOVE_CAP, predictor, C, KOMI, DETERMINISTIC_MOVES)



