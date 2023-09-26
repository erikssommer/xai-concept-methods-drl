from utils import config
from topp import Tournament

if __name__ == '__main__':
    # Initialize the Tournament of Progressive Policies (TOPP)
    topp = Tournament(config.nr_of_anets, config.nr_of_topp_games, config.render)

    # Add the agents to the tournament
    topp.add_agents()

    # Run the tournament
    topp.run_tournament()

    # Get the results
    topp.get_results()