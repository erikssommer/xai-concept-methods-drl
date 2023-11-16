import env
import os
from policy import ActorCriticNet, ResNet
from utils import config

if __name__ == "__main__":

    """
    Play against the best model
    """

    board_size = config.board_size

    go_env = env.GoEnv(size=board_size)

    go_env.reset()

    # Find the model with the highest number in the name from the models/board_size_5 folder
    path = f'../models/play_against/board_size_{board_size}/'

    folders = os.listdir(path)

    # Sort the folders by the number in the name
    sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1].strip('.keras')))

    # Get the last folder
    path = path + sorted_folders[-1]

    if config.resnet:
        actor_net = ResNet(config.board_size, path)
    else:
        actor_net = ActorCriticNet(config.board_size, path)
        
    greedy_move = True

    games = 1
    winns = 0

    start_player = input("Do you want to start? (y/n): ")

    # Test if the user typed either y or n
    while start_player not in ["y", "n"]:
        print("Invalid input, try again")
        start_player = input("Do you want to start? (y/n): ")

    for _ in range(games):
        go_env.reset()

        game_over = False

        while not game_over:
            # Get the value estimation of the current state
            value = actor_net.value_estimation(go_env.state())
            
            print(f"Value estimation of the current state: {value}")

            if go_env.turn() == 0 and start_player == "n":
                action = actor_net.best_action(go_env.state(), greedy_move)
                _, _, game_over, _ = go_env.step(action)
            elif go_env.turn() == 1 and start_player == "y":
                action = actor_net.best_action(go_env.state(), greedy_move)
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
                user_input_action = user_input_action[0] * config.board_size + user_input_action[1]

                valid_moves = go_env.valid_moves()

                # Test if the action is valid
                while user_input_action > len(valid_moves) or valid_moves[user_input_action] == 0:
                    print("Invalid action, try again")
                    user_input_action = tuple(int(n) for n in input("Enter action, eks: 2,1 or enter for pass: ").split(","))

                _, _, game_over, _ = go_env.step(user_input_action)
        
        winner = go_env.winning()

        if winner == 1:
            print("Black won")
        elif winner == -1:
            print("White won")
        else:
            print("Draw")

        go_env.render()