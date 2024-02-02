from topp import Topp

if __name__ == '__main__':
    number_of_games = 2
    dirs = ['saved_sessions', 'model_performance', 'training']
    version = ['endeavor', 'falcon', 'starship']
    greedy_move = True
    board_size = 5
    render = True
    canonical = True
    convnet = True

    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = Topp(board_size, number_of_games, render=render, komi=0.5, dir=dirs[2], version=version[2], canonical=canonical)

    # Add the agents to the tournament
    topp.add_agents(greedy_move, convnet)

    # Run the tournament
    topp.run_tournament()

    # Get the results
    topp.get_results()

    # Plot the results
    #topp.plot_results()