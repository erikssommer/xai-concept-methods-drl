from utils import config
from topp import Topp

if __name__ == '__main__':
    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = Topp(config.nr_of_topp_games, config.render)

    # Add the agents to the tournament
    topp.add_agents()

    # Run the tournament
    topp.run_tournament()

    # Get the results
    topp.get_results()

    # Plot the results
    if config.plot:
        topp.plot_results(True)