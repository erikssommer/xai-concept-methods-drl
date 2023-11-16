# Interface class for neural network policy

class BaseNet:
    def __init__(self):
        pass

    def get_all_activation_values(self, boards, keyword):
        raise NotImplementedError("Subclass must implement get_all_activation_values method")

    def fit(self, states, distributions, values, callbacks, epochs):
        raise NotImplementedError("Subclass must implement fit method")

    def predict(self, state, value_only):
        raise NotImplementedError("Subclass must implement predict method")
    
    def mask_invalid_moves(self, policy, state):
        raise NotImplementedError("Subclass must implement mask_invalid_moves method")
    
    def best_action(self, state, greedy_move, alpha):
        raise NotImplementedError("Subclass must implement best_action method")
    
    def value_estimation(self, state):
        raise NotImplementedError("Subclass must implement value_estimation method")
    
    def save_model(self, path):
        raise NotImplementedError("Subclass must implement save_model method")
    

    