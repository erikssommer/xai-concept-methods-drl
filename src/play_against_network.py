import env
from policy import ActorCriticNet
import numpy as np

if __name__ == "__main__":
    go_env = env.GoEnv(size=5)

    go_env.reset()

    path = '../models/board_size_5/net_10.keras'
    actor_net = ActorCriticNet(5, path)

    games = 10
    winns = 0

    for _ in range(games):
        go_env.reset()

        game_over = False

        while not game_over:
            if go_env.turn() == 0:
                distribution, _ = actor_net.predict(go_env.state())
                action = np.argmax(distribution[0])
                print(action)
                _, _, game_over, _ = go_env.step(action)
            else:
                go_env.render()
                user_input_action = tuple(int(n) for n in input("Enter action: ").split(","))
                _, _, game_over, _ = go_env.step(user_input_action)
        
        winner = go_env.winning()
        print(winner)
        go_env.render()