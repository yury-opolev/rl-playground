import random
import numpy as np
from game.agents.base_agent import TicTacToeAgent
from game.env import Game

class AIAgent(TicTacToeAgent):
    def __init__(self, player_token, td_model):
        self.player_token = player_token
        self.td_model = td_model
        self.name = 'AI'

    def get_action(self, actions, game, epsilon=0.0):
        v_best = None
        a_best = None
        
        if not actions:
            return (a_best, 0.0)

        features = game.extract_features()
        if np.random.binomial(1, epsilon) != 0:
            random_action = random.choice(list(actions))
            random_action_value = self.td_model.get_output(features, random_action)
            return (random_action, random_action_value)

        for a in actions:
            v = self.td_model.get_output(features, a)

            if self.player_token != Game.TOKEN_X:
                v = -1.0 * v

            if (v_best == None) or (v > v_best):
                v_best = v
                a_best = a

        # return action and it's value
        return (a_best, v_best)
