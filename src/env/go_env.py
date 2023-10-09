from enum import Enum
import numpy as np

from . import govars
from . import gogame

"""
Code from cloned from: https://github.com/aigagror/GymGo
Some ajustments has been made to fit the project
"""

class RewardMethod(Enum):
    """
    REAL: 0 = game is ongoing, 1 = black won, -1 = game tied or white won
    HEURISTIC: If game is ongoing, the reward is the area difference between black and white.
    Otherwise the game has ended, and if black has more area, the reward is BOARD_SIZE**2, otherwise it's -BOARD_SIZE**2
    """
    REAL = 'real'
    HEURISTIC = 'heuristic'


class GoEnv():
    metadata = {'render.modes': ['terminal', 'human']}
    govars = govars
    gogame = gogame

    def __init__(self, size, komi=0, reward_method='real'):
        '''
        @param reward_method: either 'heuristic' or 'real'
        heuristic: gives # black pieces - # white pieces.
        real: gives 0 for in-game move, 1 for winning, -1 for losing,
            0 for draw, all from black player's perspective
        '''
        self.size = size
        self.komi = komi
        self.state_ = gogame.init_state(size)
        self.reward_method = RewardMethod(reward_method)
        self.observation_space = (govars.NUM_CHNLS, size, size)
        self.action_space = gogame.action_size(self.state_)
        self.done = False

    def reset(self):
        '''
        Reset state, go_board, curr_player, prev_player_passed,
        done, return state
        '''
        self.state_ = gogame.init_state(self.size)
        self.done = False
        return np.copy(self.state_)

    def step(self, action):
        '''
        Assumes the correct player is making a move. Black goes first.
        return observation, reward, done, info
        '''
        assert not self.done
        if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, np.ndarray):
            assert 0 <= action[0] < self.size
            assert 0 <= action[1] < self.size
            action = self.size * action[0] + action[1]
        elif action is None:
            action = self.size ** 2

        self.state_ = gogame.next_state(self.state_, action, canonical=False)
        self.done = gogame.game_ended(self.state_)
        return np.copy(self.state_), self.reward(), self.done, self.info()

    def game_ended(self):
        return self.done

    def turn(self):
        return gogame.turn(self.state_)

    def prev_player_passed(self):
        return gogame.prev_player_passed(self.state_)

    def valid_moves(self):
        return gogame.valid_moves(self.state_)

    def uniform_random_action(self):
        valid_moves = self.valid_moves()
        valid_move_idcs = np.argwhere(valid_moves).flatten()
        return np.random.choice(valid_move_idcs)

    def info(self):
        """
        :return: Debugging info for the state
        """
        return {
            'turn': gogame.turn(self.state_),
            'invalid_moves': gogame.invalid_moves(self.state_),
            'prev_player_passed': gogame.prev_player_passed(self.state_),
        }

    def state(self):
        """
        :return: copy of state
        """
        return np.copy(self.state_)

    def canonical_state(self):
        """
        :return: canonical shallow copy of state
        """
        return gogame.canonical_form(self.state_)

    def children(self, canonical=False, padded=True):
        """
        :return: Same as get_children, but in canonical form
        """
        return gogame.children(self.state_, canonical, padded)

    def winning(self):
        """
        :return: Who's currently winning in BLACK's perspective, regardless if the game is over
        """
        return gogame.winning(self.state_, self.komi)

    def winner(self):
        """
        Get's the winner in BLACK's perspective
        :return:
        """

        if self.game_ended():
            return self.winning()
        else:
            return 0

    def reward(self):
        '''
        Return reward based on reward_method.
        heuristic: black total area - white total area
        real: 0 for in-game move, 1 for winning, 0 for losing,
            0.5 for draw, from black player's perspective.
            Winning and losing based on the Area rule
            Also known as Trump Taylor Scoring
        Area rule definition: https://en.wikipedia.org/wiki/Rules_of_Go#End
        '''
        if self.reward_method == RewardMethod.REAL:
            return self.winner()

        elif self.reward_method == RewardMethod.HEURISTIC:
            black_area, white_area = gogame.areas(self.state_)
            area_difference = black_area - white_area
            komi_correction = area_difference - self.komi
            if self.game_ended():
                return (1 if komi_correction > 0 else -1) * self.size ** 2
            return komi_correction
        else:
            raise Exception("Unknown Reward Method")

    def __str__(self):
        return gogame.str(self.state_)

    def render(self, mode='terminal'):
        if mode == 'terminal':
            print(self.__str__())
        else:
            raise NotImplementedError("Only terminal rendering is implemented")
