import os
import env
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .agent import Agent
import random
import numpy as np


class Topp:
    def __init__(self,
                 board_size: int,
                 num_games: int,
                 render: bool = False,
                 komi: float = 0.5,
                 dir: str = 'training',
                 version: str = None,
                 model_type: str = 'convnet'):

        self.board_size = board_size
        self.num_nn = 0
        self.num_games = num_games
        self.render = render
        self.agents: Agent = []
        self.move_cap = 100
        self.dir = dir
        self.version = version
        self.komi = komi
        self.model_type = model_type

    def add_agents(self, greedy_move: bool = False, resnet: bool = False, use_fast_predictor: bool = False) -> None:
        if self.dir in 'saved_sessions' or 'model_performance' in self.dir:
            path = f'../models/{self.dir}/{self.model_type}/board_size_{self.board_size}/{self.version}'
        else:
            path = f'../models/{self.dir}/{self.model_type}/board_size_{self.board_size}'

        print(f"Loading agents from {path}")

        folders = os.listdir(path)

        # Sort the folders by the number in the name
        sorted_folders = sorted(folders, key=lambda x: int(
            x.split('_')[-1].strip('.keras')))

        # Add the agents
        for folder in sorted_folders:
            self.agents.append(
                Agent(self.board_size, 
                      path, 
                      folder, 
                      greedy_move, 
                      resnet=resnet, 
                      use_fast_predictor=use_fast_predictor))
            
            self.num_nn += 1

        if len(self.agents) == 0:
            raise Exception("No agents found")

    def run_tournament(self) -> None:
        for i in range(self.num_nn):
            for j in range(i+1, self.num_nn):
                # Starting agent plays as black
                starting_agent = random.choice([i, j])

                if self.render:
                    # Print playing agents
                    print(
                        f'Playing agents: {self.agents[i].name} vs {self.agents[j].name}')
                    print(
                        f"Starting agent: {self.agents[starting_agent].name}")

                # Play the games
                for _ in range(self.num_games):
                    # Create the environment
                    go_env = env.GoEnv(size=self.board_size, komi=self.komi)

                    # Reset the environment
                    go_env.reset()

                    current_player = 0

                    # Track the number of times as black and white
                    # Starting player is black
                    if starting_agent == i:
                        self.agents[i].player_black += 1
                        self.agents[j].player_white += 1
                    else:
                        self.agents[j].player_black += 1
                        self.agents[i].player_white += 1

                    current_agent = starting_agent

                    # Play a random game
                    terminated = False
                    moves = 0

                    prev_turn_state = np.zeros((self.board_size, self.board_size))
                    temp_prev_turn_state = np.zeros((self.board_size, self.board_size))
                    prev_opposing_state = np.zeros((self.board_size, self.board_size))

                    # Play a game until termination
                    while not terminated:
                        curr_state = go_env.canonical_state()
                        valid_moves = go_env.valid_moves()
                        state = np.array([curr_state[0], prev_turn_state, curr_state[1], prev_opposing_state, np.full(
                                (self.board_size, self.board_size), current_player)])

                        if moves > self.move_cap:
                            print("Move cap reached in game between {} and {}, termination game!".format(
                                self.agents[i].name, self.agents[j].name))
                            sleep(1)
                            break

                        agent: Agent = self.agents[current_agent]

                        action, value_estimate = agent.choose_action(
                            state, valid_moves)

                        if self.render:
                            print(f"Agent {agent.name} chose action {action}")
                            print(
                                f"Value estimate of the current state: {value_estimate}")

                        _, _, terminated, _ = go_env.step(action)
                        moves += 1

                        # Render the board
                        if self.render:
                            go_env.render()

                        if current_agent == i:
                            current_agent = j
                        else:
                            current_agent = i

                        current_player = 1 - current_player
                        # Update the previous state
                        prev_turn_state = temp_prev_turn_state
                        prev_opposing_state = curr_state[0]
                        temp_prev_turn_state = prev_opposing_state

                    # Winner in perspective of the starting agent, 1 if won, -1 if lost, 0 if draw
                    winner = go_env.winner()

                    if self.render:
                        # Print the winner
                        print(f"Winner: {winner}")

                    # Add the score
                    if starting_agent == i and winner == 1:
                        self.agents[i].add_win(1)
                        self.agents[j].add_loss(2)
                    elif starting_agent == i and winner == -1:
                        self.agents[i].add_loss(1)
                        self.agents[j].add_win(2)
                    elif starting_agent == j and winner == 1:
                        self.agents[j].add_win(1)
                        self.agents[i].add_loss(2)
                    elif starting_agent == j and winner == -1:
                        self.agents[j].add_loss(1)
                        self.agents[i].add_win(2)
                    else:
                        self.agents[i].add_draw()
                        self.agents[j].add_draw()

                    # Swap the starting agent
                    if starting_agent == i:
                        starting_agent = j
                    else:
                        starting_agent = i

    def plot_results(self) -> None:
        # x is agent name
        x = [agent.name for agent in self.agents]
        # y is number of wins
        y = [agent.win for agent in self.agents]
        z_1 = [agent.black_win for agent in self.agents]
        z_2 = [agent.white_win for agent in self.agents]

        d = {'Agent': x*3, 'Wins': z_1 + z_2 + y,
             'Player': ['Black']*len(x) + ['White']*len(x) + ['Total']*len(x)}
        df = pd.DataFrame(data=d)
        # Set a larger width
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Agent', y='Wins', hue='Player', data=df)
        plt.title('Wins by agent')
        plt.show()

    def get_results(self) -> None:
        agents_result = sorted(self.agents, key=lambda x: x.win, reverse=True)

        for agent in agents_result:
            print(
                f"Agent {agent.name} won {agent.win} times where {agent.black_win} as black and {agent.white_win} as white, \
                  lost {agent.loss} times, where {agent.black_loss} as black and {agent.white_loss} as white, \
                  and drew {agent.draw} times, \
                  played {agent.player_black} times as black and {agent.player_white} times as white")
