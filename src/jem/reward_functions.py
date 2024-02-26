from .concepts import concept_functions_to_use
from abc import ABC, abstractmethod

class RewardFunction(ABC):
    @abstractmethod
    def reward_function(self, board_state, outcome):
        pass

    @staticmethod
    def get_reward_function(reward_function_name):
        if reward_function_name == "zero_sum":
            return ZeroSumRewardFunction()
        elif reward_function_name == "concept_fn":
            return ConceptRewardFunction()
        elif reward_function_name == "jem":
            return JemRewardFunction()
        else:
            raise ValueError(f"Invalid reward function name: {reward_function_name}")

class ZeroSumRewardFunction(RewardFunction):
    def reward_function(self, board_state, outcome):
        return outcome

class ConceptRewardFunction(RewardFunction):
    def reward_function(self, board_state, outcome):
        for concept_function in concept_functions_to_use():
            if concept_function.__name__ == 'null':
                continue
            presence, _, reward = concept_function(board_state, reward_shaping=True)
            if presence:
                # Return the reward pluss the outcome and maximum 1
                return min(1, reward + outcome)

        return outcome

class JemRewardFunction(RewardFunction):
    def reward_function(self, board_state, outcome):
        # Throw not implemented error
        raise NotImplementedError("Jem reward function not implemented")