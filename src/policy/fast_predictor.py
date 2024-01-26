from .lite_model import LiteModel
import numpy as np
from env import gogame
import utils

class FastPredictor:
    def __init__(self, model: LiteModel):
        self.model = model

    def predict(self, board, player, valid_moves):
        state = np.delete(board, [3,5], axis=0)
        if player == 1:
            state[2] = np.ones((board.shape[1], board.shape[2]))

        res = self.model.predict_single(state)

        policy, value = res

        policy = self.mask_invalid_moves(policy, valid_moves)

        return policy, value[0]
    
    def mask_invalid_moves(self, policy, valid_moves):

        # Mask the invalid moves
        policy = policy * valid_moves

        # Reduce to 8 decimal places
        policy = np.round(policy, 8)
    
        # Normalize the policy
        policy = utils.normalize(policy)
        
        return policy