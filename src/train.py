from rl import rl, rl_multiprocessing
from utils import Timer, folder_setup
from utils import config
    
# Main method
if __name__ == "__main__":
    folder_setup()
    
    # Start a timer
    timer = Timer()
    timer.start_timer()

    # Train the models with reinforcement learning
    if config.multi_process:
        rl_multiprocessing()
    else:
        rl()

    # End the timer
    timer.end_timer()