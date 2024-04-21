import random

class AIAgent(object):
    def __init__(self, player_token, ai_model):
        self.player_token = player_token
        self.ai_model = ai_model
        self.name = 'AI'

    def get_action(self, actions, game=None):
        v_best = 0
        a_best = None

        for a in actions:
            game.take_action(a, self.player_token)
            features = game.extract_features(self.player_token)
            v = self.ai_model.get_output(features)
            v = (1.0 - v) # 'reversing' action value to get worst state for an opponent
            if v > v_best:
                v_best = v
                a_best = a

            game.undo_action(a, self.player_token)

        return a_best
