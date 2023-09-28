def mcts_threading(args):
    print(f"Thread {args[0]} started")
    
    thread, episodes, epsilon, sigma, c, simulations, board_size = args
    
