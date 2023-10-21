import unittest

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

import env

import concepts

class TestConcepts(unittest.TestCase):
    def test_concept_win_on_pass(self):

        # Game state where previous move was a pass, and black has more area, 4th array is only ones
        game_state = None

        self.assertTrue(concepts.concept_win_on_pass(game_state))

        # Game state where previous move was a pass, and white has more area, 4th array is only ones
        game_state = None

        self.assertFalse(concepts.concept_win_on_pass(game_state))

        # Game state where previous move was not a pass, and black has more area, 4th array is only ones
        game_state = None

        self.assertFalse(concepts.concept_win_on_pass(game_state))

        # Game state where previous move was not a pass, and white has more area, 4th array is only ones
        game_state = None

        self.assertFalse(concepts.concept_win_on_pass(game_state))
    
    def test_concept_eye(self):
        game_state = [
                [[1., 0., 1., 0., 0.],
                [0., 1., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.]],

                [[1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],
            ]
        
        self.assertTrue(concepts.concept_eye(game_state))


    def test_concept_two_eyes(self):
        game_state = [
                [[0., 1., 0., 1., 0.],
                [1., 1., 1., 1., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.]],

                [[1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],

                [[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0.]],
            ]
        
        self.assertTrue(concepts.concept_two_eyes(game_state))

    
    def test_concept_with_play(self):
        go_env = env.GoEnv(5)

        CONCEPT_FUNC = concepts.concept_eye

        game_states_with_concept = []
        game_states_without_concept = []

        game_over = False

        while not game_over:
            action = go_env.uniform_random_action()
            _, _, game_over, _ = go_env.step(action)

            if CONCEPT_FUNC(go_env.state()):
                game_states_with_concept.append(go_env.state())
            else:
                game_states_without_concept.append(go_env.state())
        
        print(len(game_states_without_concept))
        print(len(game_states_with_concept))
        print(game_states_with_concept)

        assert len(game_states_with_concept) < len(game_states_without_concept)



if __name__ == '__main__':
    unittest.main()