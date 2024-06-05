import random
from game.agents.base_agent import TicTacToeAgent

class RandomAgent(TicTacToeAgent):
    def __init__(self, player_token):
        self.player_token = player_token
        self.name = 'Random'

    def get_action(self, actions, game=None, epsilon=0.0):
        if not actions:
            return None, None
        
        return random.choice(list(actions)), None
