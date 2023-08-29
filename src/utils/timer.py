import time
from datetime import datetime
from utils.read_config import config

# Class for timing the training


class Timer:

    def __init__(self):
        self.fmt = '%H:%M:%S'

    def start_timer(self):
        self.start_datetime = time.strftime(self.fmt)

    def end_timer(self):
        self.end_datetime = time.strftime(self.fmt)
        # Calculate the time difference
        total_datetime = datetime.strptime(
            self.end_datetime, self.fmt) - datetime.strptime(self.start_datetime, self.fmt)
        print(
            f"Played {config.episodes} {config.game} games with {config.simulations} simulations per move")
        print(
            f"Started: {self.start_datetime}\nFinished: {self.end_datetime}\nTotal: {total_datetime}")