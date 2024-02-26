from .concepts import concept_functions_to_use
from abc import ABC, abstractmethod
import data_utils
from .model import JointEmbeddingModel
import numpy as np
from .explanations import Explanations

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
        jem = JointEmbeddingModel('../models/jem/joint_embedding_model.keras')

        explanation_list = data_utils.get_explanation_list()  # Get the list of explanations
        vocab, _, max_sent_len = data_utils.gen_vocab_explanations_max_len(explanation_list)

        l2_norm_arr = []  # List to store L2 norms

        total_state_embeddings = []  # List to store total state embeddings
        total_explanation_embeddings = []  # List to store total explanation embeddings

        for _, explanation in enumerate(explanation_list):
            encoded_explanation = data_utils.convert_explanation_to_integers(explanation, vocab, max_sent_len)

            state_embed, exp_embed, _ = jem.predict(board_state, encoded_explanation)
            total_state_embeddings.append(state_embed)
            total_explanation_embeddings.append(exp_embed)

            # Calculate the L2 norm
            differences = state_embed - exp_embed
            l2_norm = np.linalg.norm(differences, axis=1, ord=2)
            l2_norm_arr.append(l2_norm)
        
        predicted_index = np.argmin(np.array(l2_norm_arr))  # Get the index of the predicted explanation

        # Get the reward
        presence, explanation, reward = concept_functions_to_use()[predicted_index](board_state, reward_shaping=True)
        if presence:
            return min(1, reward + outcome)
        else:
            return outcome