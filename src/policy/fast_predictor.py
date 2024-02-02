from .lite_model import LiteModel
import utils

class FastPredictor:
    def __init__(self, model: LiteModel):
        self.model = model

    def predict(self, state, valid_moves):

        res = self.model.predict_single(state)

        policy, value = res

        policy = self.mask_invalid_moves(policy, valid_moves)

        return policy, value[0]
    
    def mask_invalid_moves(self, policy, valid_moves):

        # Mask the invalid moves
        policy = policy * valid_moves

        # Normalize the policy
        policy = utils.normalize(policy)
        
        return policy