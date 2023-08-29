from tqdm import tqdm
from utils.read_config import config

class RL:
    
    def learn(self):
        # Loop through the number of episodes
        for episode in tqdm(range(config.episodes)):
            # Play a game
            self.play_game()
            