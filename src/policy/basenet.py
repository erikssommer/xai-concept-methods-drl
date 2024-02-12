# Interface class for neural network policy
from abc import ABC, abstractmethod

class BaseNet(ABC):
    
    @abstractmethod
    def get_all_activation_values(self, boards, keyword):
        pass

    @abstractmethod
    def fit(self, states, distributions, values, callbacks, epochs):
        pass

    @abstractmethod
    def predict(self, state, value_only):
        pass
    
    @abstractmethod
    def mask_invalid_moves(self, policy, state):
        pass
    
    @abstractmethod
    def best_action(self, state, greedy_move, alpha):
        pass
    
    @abstractmethod
    def value_estimation(self, state):
        pass
    
    @abstractmethod
    def save_model(self, path):
        pass
    

    