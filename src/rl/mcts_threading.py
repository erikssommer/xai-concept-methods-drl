import tensorflow as tf
from mcts import MCTS
import env
import gc

def mcts_threading(args):
    thread, model_name, episodes, epsilon, sigma, move_cap, c, simulations, board_size = args

    state_buffer = []
    observation_buffer = []
    value_buffer = []

    go_env = env.GoEnv(size=board_size)

    if epsilon != 1:
        policy_nn = tf.keras.models.load_model(f'../models/board_size_{board_size}/net_{model_name}.keras')
    else:
        policy_nn = None

    for _ in range(episodes):
        states = []
        distributions = []
        players = []

        go_env.reset()

        game_state = go_env.canonical_state()

        tree = MCTS(game_state, epsilon, sigma, simulations, board_size, move_cap, c, policy_nn)
        
        game_over = False

        while not game_over:
            current_player = go_env.turn()

            best_action_node, game_state, distribution = tree.search()

            # Store the data
            states.append(game_state)
            distributions.append(distribution)
            players.append(current_player)

            _, _, game_over, _ = go_env.step(best_action_node.action)

            tree.set_root_node(best_action_node)

            gc.collect()

        winner = go_env.winning()

        for (state, distribution, player) in zip(states, distributions, players):

            value_map = {
                (env.govars.BLACK, 1): 1,
                (env.govars.WHITE, -1): 1,
                (env.govars.BLACK, -1): -1,
                (env.govars.WHITE, 1): -1
            }

            value = value_map.get((player, winner), 0)
                
            state_buffer.append(state)
            observation_buffer.append(distribution)
            value_buffer.append(value)

    return state_buffer, observation_buffer, value_buffer