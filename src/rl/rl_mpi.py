from typing import Tuple
import tensorflow as tf
import absl.logging
from mpi4py import MPI
import gc
import numpy as np
from utils import tensorboard_setup, write_to_tensorboard, folder_setup, Timer
from tqdm import tqdm
import env
from mcts import MCTS
from utils import config
from policy import FastPredictor, LiteModel, get_policy
from jem import data_utils
from .reward_functions import get_reward_function, RewardFunction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


absl.logging.set_verbosity(absl.logging.ERROR)


def perform_mcts_episodes(episodes: int,
                          fast_predictor_path: str,
                          simulations: int,
                          sample_ratio: float,
                          c: float,
                          komi: float,
                          board_size: int,
                          non_det_moves: int,
                          move_cap: int,
                          model_type: str,
                          reward_fn: RewardFunction) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    np.seterr(over="ignore", invalid="raise")

    state_buffer = []
    concept_buffer = []
    distribution_buffer = []
    value_buffer = []
    game_winners = []

    agent = FastPredictor(LiteModel.from_file(
        f"{fast_predictor_path}/temp.tflite"))

    for _ in range(episodes):

        turns = []
        states = []
        states_after_action = []
        distributions = []

        # Create the environment
        go_env = env.GoEnv(size=board_size, komi=komi)

        # Reset the environment
        go_env.reset()

        # Number of moves in the game
        move_nr = 0

        # Get the initial state
        init_state = go_env.canonical_state()

        # Create the initial tree
        tree = MCTS(init_state, simulations, board_size,
                    move_cap, agent, c, komi, non_det_moves)

        # Play a game until termination
        game_over = False

        # Black always starts
        curr_player = 0

        prev_turn_state = np.zeros((board_size, board_size))
        temp_prev_turn_state = np.zeros((board_size, board_size))
        prev_opposing_state = np.zeros((board_size, board_size))

        while not game_over and move_nr < move_cap:
            # Get the player
            curr_state = go_env.canonical_state()

            assert curr_state.all() == tree.root.state.all()

            best_action_node, distribution = tree.search(move_nr)

            # Apply the action to the environment
            _, _, game_over, _ = go_env.step(best_action_node.action)

            # Add the case to the replay buffer
            if np.random.random() < sample_ratio:

                state_after_action = go_env.canonical_state()

                state = np.array([curr_state[0], prev_turn_state, curr_state[1], prev_opposing_state, np.full((board_size, board_size), curr_player)])
                state_after_action = np.array([state_after_action[1], curr_state[0], state_after_action[0], curr_state[1], np.full((board_size, board_size), curr_player)])

                # Add the case to the replay buffer
                turns.append(curr_player)
                states.append(state)
                states_after_action.append(state_after_action)
                distributions.append(distribution)

            # Update the root node of the mcts tree
            tree.set_root_node(best_action_node)

            # Flipp the player
            curr_player = 1 - curr_player

            # Update the previous state
            prev_turn_state = temp_prev_turn_state
            prev_opposing_state = curr_state[0]
            temp_prev_turn_state = prev_opposing_state

            # Increment the move number
            move_nr += 1

            # Garbage collection
            gc.collect()

        # Get the winner of the game in black's perspective (1 for win and -1 for loss)
        winner = go_env.winning()

        # Do not allow draws
        assert winner != 0

        # Set the values of the states
        for (dist, state, state_after_action, turn) in zip(distributions, states, states_after_action, turns):
            if turn == 0 and winner == 1:
                outcome = 1
            elif turn == 1 and winner == -1:
                outcome = 1
            elif turn == 0 and winner == -1:
                outcome = -1
            elif turn == 1 and winner == 1:
                outcome = -1
            else:
                AssertionError("Invalid winner")

            state_buffer.append(state)
            distribution_buffer.append(dist)
            value_buffer.append(reward_fn(state_after_action, outcome))

            # Target for the concept bottleneck outputlayer
            if model_type == "conceptnet":
                concept = data_utils.binary_encode_concepts(state_after_action, outcome)
                concept_buffer.append(concept)
            
        game_winners.append(winner)

        # Delete references and garbadge collection
        del tree.root
        del tree
        del go_env

        gc.collect()

    return state_buffer, concept_buffer, distribution_buffer, value_buffer, game_winners


def rl_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranksize = comm.Get_size()

    np.seterr(over="ignore")

    board_size = config.board_size
    epochs = config.epochs
    simulations = config.simulations
    replay_buffer_cap = config.rbuf_cap
    sample_ratio = config.sample_ratio
    number_of_threads_generating_data = ranksize - 1
    episodes_per_thread_instance = config.episodes_per_epoch
    epochs_skip = config.epoch_skip
    komi = config.komi
    move_cap = config.move_cap
    cpuct = config.c
    non_deterministic_moves = config.non_det_moves
    save_intervals = config.save_intervals
    model_type = config.model_type
    reward_function_type = config.reward_function
    clear_tensorboard = config.clear_tensorboard
    fast_predictor_path = f"../models/fastpred/training/{model_type}/board_size_{board_size}"
    number_of_concepts = data_utils.get_number_of_concepts()

    if rank == 0:
        # Start a timer
        timer = Timer()
        timer.start_timer()

        save_path = folder_setup(model_type, reward_function_type, board_size)

        gpus = tf.config.list_physical_devices("GPU")
        print("\nNum GPUs Available: ", len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Initialize the buffers
        state_buffer = []
        concept_buffer = []
        distribution_buffer = []
        value_buffer = []
    else:
        reward_fn = get_reward_function(reward_function_type)

    os.makedirs(fast_predictor_path, exist_ok=True)

    if rank == 0:
        # Create the tensorboard callback
        tb_writer, tb_callback = tensorboard_setup(model_type, reward_function_type, clear_tensorboard)

        agent = get_policy(model_type, board_size, number_of_concepts)
        agent.save_model(f"{save_path}/net_0.keras")

        # Calculate the number of games in total
        total = number_of_threads_generating_data * \
            episodes_per_thread_instance * epochs

        print(f"Saving interval: {save_intervals}", flush=True)
        print(
            f"Number of threads generating data: {number_of_threads_generating_data}", flush=True)
        print(
            f"Episodes per thread instance: {episodes_per_thread_instance}", flush=True)
        print(f"Total number of games: {total}", flush=True)

    for epoch in tqdm(range(1, epochs + 1)):
        if rank == 0:
            with open(f"{fast_predictor_path}/temp.tflite", "wb") as f:
                lite_model = LiteModel.from_keras_model_as_bytes(agent.model)
                f.write(lite_model)

        comm.Barrier()
        if rank != 0:
            results = perform_mcts_episodes(
                episodes_per_thread_instance,
                fast_predictor_path,
                simulations,
                sample_ratio,
                cpuct,
                komi,
                board_size,
                non_deterministic_moves,
                move_cap,
                model_type,
                reward_fn
            )

            data = {
                "states": results[0],
                "concepts": results[1],
                "distributions": results[2],
                "values": results[3],
                "winners": results[4]
            }
            # print("Results from {} sent.".format(rank), flush=True)
            comm.send(data, 0)
            del data
            del results
        else:
            outcomes = []
            for _ in range(1, ranksize):
                results = comm.recv()
                # print("A result is recieved!", flush=True)
                state_buffer.extend(results["states"])
                concept_buffer.extend(results["concepts"])
                distribution_buffer.extend(results["distributions"])
                value_buffer.extend(results["values"])
                outcomes.extend(results["winners"])

            outcomes = np.array(outcomes)

        # Generate stuff for three epochs before starting to checkpoint and train
        if rank != 0:
            continue
        # Now this is only used for the GPU-thread
        history = None
        if epoch > epochs_skip:
            state_buffer = state_buffer[-replay_buffer_cap:]
            concept_buffer = concept_buffer[-replay_buffer_cap:]
            distribution_buffer = distribution_buffer[-replay_buffer_cap:]
            value_buffer = value_buffer[-replay_buffer_cap:]

            if model_type == "conceptnet":
                history = agent.fit(
                    np.array(state_buffer),
                    np.array(concept_buffer),
                    np.array(distribution_buffer),
                    np.array(value_buffer),
                    epochs=1,
                    callbacks=[tb_callback]
                )
            else:
                history = agent.fit(
                    np.array(state_buffer),
                    np.array(distribution_buffer),
                    np.array(value_buffer),
                    epochs=1,
                    callbacks=[tb_callback]
                )

            # Add the metrics to TensorBoard
            write_to_tensorboard(tb_writer, history, outcomes, epoch)

        if epoch != 0 and epoch in save_intervals:
            agent.save_model(f'{save_path}/net_{epoch}.keras')

    if rank == 0:
        # End the timer
        timer.end_timer()
