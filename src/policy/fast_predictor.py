from .lite_model import LiteModel
import utils
import numpy as np
from typing import Tuple

class FastPredictor:
    def __init__(self, model: LiteModel) -> None:
        self.model = model

    def predict(self, state: np.ndarray, valid_moves: np.ndarray) -> Tuple[np.ndarray, float]:

        res = self.model.predict_single(state)

        policy, value = res

        policy = self.mask_invalid_moves(policy, valid_moves)

        return policy, value[0]
    
    def mask_invalid_moves(self, policy: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:

        # Mask the invalid moves
        policy = policy * valid_moves

        # Convert to 8 decimals
        policy = np.round(policy, 8)

        # Normalize the policy
        policy = utils.normalize(policy)
        
        return policy
    
    def best_action(self, state: np.ndarray, valid_moves: np.ndarray, greedy_move: bool = False, alpha: float = None) -> Tuple[int, float]:
        policy, value = self.predict(state, valid_moves)

        if greedy_move:
            return np.argmax(policy), value

        if alpha and np.random.random() < alpha:
            # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
            return np.argmax(policy), value

        # Selecting move randomly, but weighted by the distribution (0 = argmax, 1 = probablistic)
        return np.random.choice(len(policy), p=policy), value