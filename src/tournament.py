from topp import Topp

if __name__ == '__main__':
    number_of_games = 2
    dirs = ['saved_sessions', 'model_performance', 'training']
    version = ['endeavor', 'falcon', 'starship']
    greedy_move = True
    board_size = 5
    render = False
    resnet = False
    model_type = 'resnet' if resnet else 'convnet'

    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = Topp(board_size, number_of_games, render=render, komi=1.5, dir=dirs[0], version=version[1], model_type=model_type)

    # Add the agents to the tournament
    topp.add_agents(greedy_move, resnet)

    # Run the tournament
    topp.run_tournament()

    # Get the results
    topp.get_results()

    # Plot the results
    topp.plot_results()