import random
import numpy as np
from game.game import Game

class QTableAgent(object):
    def __init__(self, player_token, q_table):
        self.player_token = player_token
        self.q_table = q_table
        self.name = 'QTable'

    def get_action(self, actions, game=None):
        v_best = 0
        a_best = None

        for a in actions:
            game.take_action(a, self.player_token)
            potentional_state = game.extract_qstate()
            v = self.q_table[potentional_state]
            if self.player_token != Game.TOKEN_X:
                v = 1.0 - v

            if v > v_best:
                v_best = v
                a_best = a

            game.undo_action(a, self.player_token)

        return a_best


