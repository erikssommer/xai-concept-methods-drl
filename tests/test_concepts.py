import unittest

import os
# Add the src folder to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'src')))

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


if __name__ == '__main__':
    unittest.main()