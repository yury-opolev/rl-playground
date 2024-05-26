import random
from game.game import Game

class AIAgent(object):
    def __init__(self, player_token, ai_model):
        self.player_token = player_token
        self.ai_model = ai_model
        self.name = 'AI'
        self.epsilon = 0.5

    def get_action(self, actions, game=None, greedy=True):
        if not greedy and (random.random() < self.epsilon):
            return random.choice(list(actions))

        v_best = None
        a_best = None

        for a in actions:
            game.take_action(a, self.player_token)
            features = game.extract_features()
            v = self.ai_model.get_output(features)
            if self.player_token != Game.TOKEN_X:
                v = 1.0 - v

            if (v_best == None) or (v > v_best):
                v_best = v
                a_best = a

            game.undo_action(a, self.player_token)

        return a_best
