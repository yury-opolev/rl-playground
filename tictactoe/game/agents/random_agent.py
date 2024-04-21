import random

class RandomAgent(object):
    def __init__(self, player_token):
        self.player_token = player_token
        self.name = 'Random'

    def get_action(self, moves, game=None):
        return random.choice(list(moves)) if moves else None
