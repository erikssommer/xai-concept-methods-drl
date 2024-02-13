import env
import os
from policy import ConvNet, ResNet, FastPredictor, LiteModel
from utils import config
from mcts import MCTS
import numpy as np

def play_game():
    """
    Play against the best model
    """

    board_size = config.board_size
    simulations = config.simulations
    move_cap = board_size ** 2 * 4
    c = config.c
    komi = config.komi
    det_moves = 0
    resnet = config.resnet
    greedy_move = True

    model_type = "resnet" if resnet else "convnet"

    go_env = env.GoEnv(size=board_size, komi=komi)

    go_env.reset()

    # Find the model with the highest number in the name from the models/board_size_5 folder
    path = f'../models/play_against/{model_type}/board_size_{board_size}/'

    folders = os.listdir(path)

    # Sort the folders by the number in the name
    sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1].strip('.keras')))

    # Get the last folder
    path = path + sorted_folders[-1]

    if resnet:
        actor_net = ResNet(board_size, path)
    else:
        actor_net = ConvNet(board_size, path)

    model = FastPredictor(LiteModel.from_keras_model(actor_net.model))

    games = 1
    winns = 0

    start_player = input("Do you want to start? (y/n): ")

    # Test if the user typed either y or n
    while start_player not in ["y", "n"]:
        print("Invalid input, try again")
        start_player = input("Do you want to start? (y/n): ")
    
    lookahead = input("Do you want to use lookahead? (y/n): ")

    # Test if the user typed either y or n
    while lookahead not in ["y", "n"]:
        print("Invalid input, try again")
        lookahead = input("Do you want the model to use lookahead with MCTS? (y/n): ")

    for _ in range(games):
        go_env.reset()

        game_over = False

        curr_player = 0
        move_nr = 0

        prev_turn_state = np.zeros((board_size, board_size))
        temp_prev_turn_state = np.zeros((board_size, board_size))
        prev_opposing_state = np.zeros((board_size, board_size))

        while not game_over:
            game_state = go_env.canonical_state()
            
            valid_moves = go_env.valid_moves()

            if curr_player == 0:
                state = np.array([game_state[0], prev_turn_state, game_state[1], prev_opposing_state, np.zeros((board_size, board_size))])
            else:
                state = np.array([game_state[0], prev_turn_state, game_state[1], prev_opposing_state, np.ones((board_size, board_size))])
            # Get the value estimation of the current state
            value = actor_net.value_estimation(state, valid_moves)
            
            print(f"Value estimation of the current state: {value}")

            if curr_player == 0 and start_player == "n" and lookahead:
                tree = MCTS(game_state, simulations, board_size, move_cap, model, c, komi, det_moves)
                node, _ = tree.search(move_nr)
                _, _, game_over, _ = go_env.step(node.action)
            elif curr_player == 1 and start_player == "y" and lookahead:
                tree = MCTS(game_state, simulations, board_size, move_cap, model, c, komi, det_moves)
                node, _ = tree.search(move_nr)
                _, _, game_over, _ = go_env.step(node.action)
            elif curr_player == 0 and start_player == "n":
                action, _ = actor_net.best_action(state, valid_moves, greedy_move)
                _, _, game_over, _ = go_env.step(action)
            elif curr_player == 1 and start_player == "y":
                action, _ = actor_net.best_action(state, valid_moves, greedy_move)
                _, _, game_over, _ = go_env.step(action)
            else:
                go_env.render()
                user_input = input("Enter action: ")

                # Test if enter is pressed
                if user_input == "":
                    _, _, game_over, _ = go_env.step(None)
                    continue

                user_input_action = tuple(int(n) for n in user_input.split(","))
                
                # Convert it info a number
                user_input_action = user_input_action[0] * board_size + user_input_action[1]

                valid_moves = go_env.valid_moves()

                # Test if the action is valid
                while user_input_action > len(valid_moves) or valid_moves[user_input_action] == 0:
                    print("Invalid action, try again")
                    user_input_action = tuple(int(n) for n in input("Enter action, eks: 2,1 or enter for pass: ").split(","))

                _, _, game_over, _ = go_env.step(user_input_action)

            curr_player = 1 - curr_player
            move_nr += 1

            # Update the previous state
            prev_turn_state = temp_prev_turn_state
            prev_opposing_state = game_state[0]
            temp_prev_turn_state = prev_opposing_state
        
        winner = go_env.winning()

        go_env.render()

        print("Game over!")

        if winner == 1:
            print("Black won")
        else:
            print("White won")


if __name__ == "__main__":
    play_game()