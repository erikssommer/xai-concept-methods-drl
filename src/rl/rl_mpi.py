import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
from policy import ConvNet, ResNet, FastPredictor, LiteModel
from utils import config
from mcts import MCTS
import env
from tqdm import tqdm

from utils import tensorboard_setup, write_to_tensorboard, folder_setup, Timer

import numpy as np
import gc
from mpi4py import MPI
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

def perform_mcts_episodes(args):

    np.seterr(over="ignore", invalid="raise")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    episodes, fast_predictor_path, simulations, sample_ratio, c, komi, board_size, det_moves, move_cap, thread = args

    state_buffer = []
    distribution_buffer = []
    value_buffer = []
    game_winners = []

    agent = FastPredictor(LiteModel.from_file(f"{fast_predictor_path}/temp.tflite"))

    for episode in range(episodes):
        #print("Thread {} starting, episode {}.".format(thread, episode + 1), flush=True)

        turns = []
        states = []
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
        tree = MCTS(init_state, simulations, board_size, move_cap, agent, c, komi, det_moves)

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

            # Add the case to the replay buffer
            if np.random.random() < sample_ratio:
                if curr_player == 0:
                    state = np.array([curr_state[0], prev_turn_state, curr_state[1], prev_opposing_state, np.zeros((board_size, board_size))])
                else:
                    state = np.array([curr_state[0], prev_turn_state, curr_state[1], prev_opposing_state, np.ones((board_size, board_size))])

                # Add the case to the replay buffer
                turns.append(curr_player)
                states.append(state)
                distributions.append(distribution)

            # Apply the action to the environment
            _, _, game_over, _ = go_env.step(best_action_node.action)

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
        for (dist, state, turn) in zip(distributions, states, turns):
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
            value_buffer.append(outcome)
        
        game_winners.append(winner)

        # Delete references and garbadge collection      
        del tree.root
        del tree
        del go_env

        gc.collect()

    # print("Thread {} finished.".format(thread))
    
    return state_buffer, distribution_buffer, value_buffer, game_winners


def rl_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ranksize = comm.Get_size()
    #print("Rank {} of {} started.".format(rank, ranksize), flush=True)
    amount_of_gpus = 1
    np.seterr(over="ignore")
    if rank == 0:
        # Start a timer
        timer = Timer()
        timer.start_timer()

        folder_setup()

        gpus = tf.config.list_physical_devices("GPU")
        print("\nNum GPUs Available: ", len(gpus))
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    BOARD_SIZE = config.board_size
    USE_RESNET = config.resnet
    EPOCHS = config.epochs
    SIM_STEPS = config.simulations
    REPLAY_BUFFER_CAP = config.rbuf_cap
    SAMPLE_RATIO = config.sample_ratio

    NUM_THEADS_GENERATING_DATA = ranksize - 1
    # Take the number of epochs into account
    EPISODES_PER_THREAD_INSTANCE = config.episodes_per_epoch

    SAVE_INTERVAL = EPOCHS // config.nr_of_anets
    EPOCHS_SKIP = config.epoch_skip

    KOMI = config.komi
    MOVE_CAP = config.move_cap
    CPUCT = config.c
    DET_MOVES = config.det_moves
    FAST_PREDICTOR_PATH = f"../models/fastpred/training/board_size_{BOARD_SIZE}"

    if rank == 0:
        state_buffer = []
        distribution_buffer = []
        value_buffer = []
        outcomes = []

    os.makedirs(FAST_PREDICTOR_PATH, exist_ok=True)

    if rank == 0:
        # Create the tensorboard callback
        tensorboard_callback, logdir = tensorboard_setup()

        if USE_RESNET:
            agent = ResNet(BOARD_SIZE)
        else:
            agent = ConvNet(BOARD_SIZE)
        agent.save_model(f"../models/training/board_size_{BOARD_SIZE}/net_0.keras")

        # Calculate the number of games in total
        total = NUM_THEADS_GENERATING_DATA * EPISODES_PER_THREAD_INSTANCE * EPOCHS

        print("Episodes per thread instance: {}".format(EPISODES_PER_THREAD_INSTANCE), flush=True)
        print("Total number of games: {}".format(total), flush=True)


    for epoch in tqdm(range(1, EPOCHS + 1)):
        if rank == 0:
            with open(f"{FAST_PREDICTOR_PATH}/temp.tflite", "wb") as f:
                lite_model = LiteModel.from_keras_model_as_bytes(agent.model)
                f.write(lite_model)

        comm.Barrier()
        if rank != 0:
            results = perform_mcts_episodes((
                EPISODES_PER_THREAD_INSTANCE,
                FAST_PREDICTOR_PATH,
                SIM_STEPS,
                SAMPLE_RATIO,
                CPUCT,
                KOMI,
                BOARD_SIZE,
                DET_MOVES,
                MOVE_CAP,
                rank
            ))

            data = {
                "states": results[0],
                "distributions": results[1],
                "values": results[2],
                "winners": results[3]
            }
            #print("Results from {} sent.".format(rank), flush=True)
            comm.send(data, 0)
            del data
            del results
        else:
            for _ in range(1, ranksize):
                results = comm.recv()
                #print("A result is recieved!", flush=True)
                state_buffer.extend(results["states"])
                distribution_buffer.extend(results["distributions"])
                value_buffer.extend(results["values"])
                outcomes.extend(results["winners"])
        # Generate stuff for three epochs before starting to checkpoint and train
        if rank != 0:
            continue
        # Now this is only used for the GPU-thread
        history = None
        if epoch > EPOCHS_SKIP:
            state_buffer = state_buffer[-REPLAY_BUFFER_CAP:]
            distribution_buffer = distribution_buffer[-REPLAY_BUFFER_CAP:]
            value_buffer = value_buffer[-REPLAY_BUFFER_CAP:]

            print(len(state_buffer))

            history = agent.fit(
                np.array(state_buffer), 
                np.array(distribution_buffer), 
                np.array(value_buffer), 
                epochs=1,
                callbacks=[tensorboard_callback]
            )

            # Add the metrics to TensorBoard
            write_to_tensorboard(history, epoch, logdir)

        if epoch != 0 and epoch % SAVE_INTERVAL == 0:
            agent.save_model(f'../models/training/board_size_{BOARD_SIZE}/net_{epoch}.keras')
    
    if rank == 0:
        # Loop through the outcomes and calculate the winrate
        winrate = 0
        for outcome in outcomes:
            if outcome == 1:
                winrate += 1
        winrate /= len(outcomes)

        print("Winrate as black: {}".format(winrate), flush=True)

        # End the timer
        timer.end_timer()