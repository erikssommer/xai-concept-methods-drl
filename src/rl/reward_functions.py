from abc import ABC, abstractmethod
import numpy as np

from jem import JointEmbeddingModel, data_utils, concept_functions_to_use

class RewardFunction(ABC):
    @abstractmethod
    def reward_function(self, board_state, outcome):
        pass

class ZeroSumRewardFunction(RewardFunction):
    """
    Zero sum reward function that uses the outcome as the reward, i.e. 1 for win, -1 for loss, sum of the rewards is 0 
    """
    def reward_function(self, board_state, outcome):
        return outcome

class ConceptRewardFunction(RewardFunction):
    """
    Reward function that uses the concept functions to find the explanation and corresponding reward
    """
    def reward_function(self, board_state, outcome):
        # Actions leading to a win is allways rewarded with 1 (max reward)
        if outcome == 1:
            return 1
        
        highest_reward = 0
        for concept_function in concept_functions_to_use():
            if concept_function.__name__ == 'null':
                continue
            presence, _, reward = concept_function(board_state, reward_shaping=True)
            if presence:
                if reward > highest_reward:
                    highest_reward = reward

        # Using the highest reward (Option is to stack the rewards and use the sum of the rewards)
        if highest_reward > 0:
            # Return the highest reward pluss the outcome and maximum 1
            return min(1, highest_reward + outcome)

        return outcome

class JemRewardFunction(RewardFunction):
    """
    Reward function that uses the joint embedding model to find the closest explanation and corresponding reward
    """
    def __init__(self):
        self.jem = JointEmbeddingModel(load_path='../models/jem/joint_embedding_model.keras')
        self.explanation_list = data_utils.get_explanation_list()
        self.vocab, _, self.max_sent_len = data_utils.gen_vocab_explanations_max_len(self.explanation_list)

    def reward_function(self, board_state, outcome):
        # Actions leading to a win is allways rewarded with 1(max reward)
        if outcome == 1:
            return 1
        
        l2_norm_arr = []  # List to store L2 norms

        total_state_embeddings = []  # List to store total state embeddings
        total_explanation_embeddings = []  # List to store total explanation embeddings

        for _, explanation in enumerate(self.explanation_list):
            encoded_explanation = data_utils.convert_explanation_to_integers(explanation, self.vocab, self.max_sent_len)

            state_embed, exp_embed = self.jem.predict(board_state, encoded_explanation)
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


def get_reward_function(reward_function_type):
    if reward_function_type == "zero_sum":
        return ZeroSumRewardFunction().reward_function
    elif reward_function_type == "concept_fn":
        return ConceptRewardFunction().reward_function
    elif reward_function_type == "jem":
        return JemRewardFunction().reward_function
    else:
        raise ValueError(f"Invalid reward function type: {reward_function_type}")