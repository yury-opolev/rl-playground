import random
import numpy as np
from game.agents.base_agent import TicTacToeAgent
from game.env import Game

class AIAgentStateVal(TicTacToeAgent):
    def __init__(self, player_token, td_model_val):
        self.player_token = player_token
        self.td_model_val = td_model_val
        self.name = 'AI(StateVal)'

    def get_action(self, actions, game, epsilon=0.0):
        v_best = None
        a_best = None
        
        if not actions:
            return (a_best, 0.0)

        if np.random.binomial(1, epsilon) != 0:
            random_action = random.choice(list(actions))
            game.take_action(random_action, game.current_player_token)
            features = game.extract_features()
            random_action_value = self.td_model_val.get_state_value(features)
            game.undo_action(random_action)
            return (random_action, random_action_value)

        for a in actions:
            game.take_action(a, game.current_player_token)
            features = game.extract_features()
            v = self.td_model_val.get_state_value(features)
            game.undo_action(a)

            if self.player_token != Game.TOKEN_X:
                v = -1.0 * v

            if (v_best == None) or (v > v_best):
                v_best = v
                a_best = a

        # return action and it's value
        if self.player_token != Game.TOKEN_X:
            v_best = -1.0 * v_best
        return (a_best, v_best)
