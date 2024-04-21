import random

class RandomAgent(object):
    def __init__(self, player_token):
        self.player_token = player_token
        self.name = 'Random'

    def get_action(self, actions, game=None):
        return random.choice(list(actions)) if actions else None
