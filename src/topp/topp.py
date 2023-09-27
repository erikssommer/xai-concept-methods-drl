import os
import env
from time import sleep
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .agent import Agent
import random
from utils import config

class Tournament:
    def __init__(self, num_nn, num_games, render: bool = False):
        self.num_nn = num_nn
        self.num_games = num_games
        self.render = render
        self.agents: Agent = []
        self.move_cap = 100

    def add_agents(self):
        path = '../models/'

        folders = os.listdir(path)

        # Sort the folders by the number in the name
        sorted_folders = sorted(folders, key=lambda x: int(x.split('_')[-1]))

        # Add the agents
        for folder in sorted_folders:
            self.agents.append(Agent(path, folder))

        if len(self.agents) == 0:
            raise Exception("No agents found")
        
    def run_tournament(self):
        invalid_moves = 0
        moves_total = 0
        for i in range(self.num_nn):
            for j in range(i+1, self.num_nn):
                # Starting agent plays as black
                starting_agent = random.choice([i, j])

                # Play the games
                for _ in range(self.num_games):
                    # Create the environment
                    go_env = env.GoEnv(size=config.board_size)

                    # Reset the environment
                    go_env.reset()

                    current_agent = starting_agent

                    # Play a random game
                    terminated = False
                    moves = 0

                    # Play a game until termination
                    while not terminated:
                        if moves > self.move_cap:
                            print("Move cap reached, terminating game")
                            sleep(1)
                            break

                        agent: Agent = self.agents[current_agent]
                        action, valid = agent.choose_action(go_env.state())

                        if not valid:
                            invalid_moves += 1
                        moves_total += 1
                        
                        _, _, terminated, _ = go_env.step(action)
                        moves += 1

                        # Render the board
                        if self.render:
                            go_env.render()
                        
                        if current_agent == i:
                            current_agent = j
                        else:
                            current_agent = i

                    # Winner in perspective of the starting agent, 1 if won, -1 if lost, 0 if draw
                    winner = go_env.winner()

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
                    starting_agent = (starting_agent + 1) % 2
        
        print("Total number of invalid moves: {}".format(invalid_moves))
        print("Total number of moves: {}".format(moves_total))

    def plot_results(self, block):
        plt.clf()
        plt.ion()
        # x is agent name
        x = [agent.name for agent in self.agents]
        # y is number of wins
        y = [agent.win for agent in self.agents]
        z_1 = [agent.player_1_win for agent in self.agents]
        z_2 = [agent.player_2_win for agent in self.agents]

        d = {'Agent': x*3, 'Wins': z_1 + z_2 + y, 'Player': ['Player 1']*len(x) + ['Player 2']*len(x) + ['Total']*len(x)}
        df = pd.DataFrame(data=d)

        sns.barplot(x='Agent', y='Wins', hue='Player', data=df)
        plt.show(block=block)

    def get_results(self):
        agents_result = sorted(self.agents, key=lambda x: x.win, reverse=True)

        for agent in agents_result:
            print(
                f"Agent {agent.name} won {agent.win} times, lost {agent.loss} times and drew {agent.draw} times")