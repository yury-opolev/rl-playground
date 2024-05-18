import random
import pickle

from game.game import Game
from game.agents.q_table_agent import QTableAgent

class QTableModel(object):
    def __init__(self, model_path, restore=False):
        self.model_path = model_path

        self.q_table = {}
        for element in Game.validBoards():
            self.q_table[str(element)] = random.uniform(0, 1)

        if restore:
            self.restore()

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def restore(self):
        with open(self.model_path, 'rb') as f:
            self.q_table = pickle.load(f)

    def get_output(self, x):
        return self.q_table[x]

    def test(self, episodes=100):
        pass

    def train_q_table(self):
        validation_interval = 500
        episodes = 5000
        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            player_agents = [QTableAgent('X', self.q_table), QTableAgent('O', self.q_table)]
            game = Game()

            player_num = random.randint(0, 1)
            game.current_player_token = game.player_tokens[player_num]

            game_step = 0
            while not game.is_finished():
                observed_state = game.extract_features()
                state_value = player_agent.ai_model.get_output(observed_state)

                player_agent = player_agents[player_num]
                game.make_move(player_agent)

                player_num = (player_num + 1) % 2
                game.current_player_token = game.player_tokens[player_num]

                next_observed_state = game.extract_features()
                if (game.is_finished()):
                    if game.winner_token() == Game.TOKEN_X:
                        next_state_value = 1.0
                    elif game.winner_token() == Game.TOKEN_O:
                        next_state_value = 0.0
                    else:
                        next_state_value = 0.5
                else:
                    next_state_value = player_agent.ai_model.get_output(next_observed_state)

                # self.sess.run(self.train_op, feed_dict={ self.x: x, next_state_value })

                game_step += 1

            winner = game.winner()