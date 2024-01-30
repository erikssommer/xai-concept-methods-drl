from rl import rl, rl_multiprocessing, rl_zero, rl_uct, rl_canonical
from utils import Timer, folder_setup
from utils import config
    
# Main method
if __name__ == "__main__":
    folder_setup()
    
    # Start a timer
    timer = Timer()
    timer.start_timer()

    if config.rl_canonical:
        rl_canonical()
    elif config.rl_zero:
        folder_setup()
        rl_zero()
    elif config.rl_uct:
        folder_setup()
        rl_uct()
    elif config.multi_process:
        folder_setup()
        rl_multiprocessing()
    else:
        folder_setup()
        rl()

    # End the timer
    timer.end_timer()