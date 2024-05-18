import random
import pickle
from pathlib import Path

from game.game import Game
from game.agents.q_table_agent import QTableAgent
from game.agents.random_agent import RandomAgent

class QTableModel(object):
    def __init__(self, model_path, restore=False, save=False):
        self.model_path = model_path

        self.q_table = {}
        for element in Game.validBoards():
            self.q_table[str(element)] = random.uniform(0, 1)

        if restore:
            self.restore()

        self.save_on_train = save

    def save(self):
        print(f"Saving to {self.model_path}")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def restore(self):
        existing_modelfile = Path(self.model_path)
        if not existing_modelfile.exists():
            return
        
        print(f"Loading from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.q_table = pickle.load(f)

    def get_output(self, x):
        return self.q_table[x]

    def test(self, episodes=100):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()
            player_agents = [QTableAgent('X', self.q_table), RandomAgent('O')]
            winner_token = game.play(player_agents, draw=False)
            if winner_token is None:
                winner_token = Game.EMPTYTOKEN
            winners[winner_token] = winners[winner_token] + 1

        print(f"Games played: {episodes}, draws: {winners[Game.EMPTYTOKEN]}, 'X' wins: {winners[Game.TOKEN_X]}, 'O' wins: {winners[Game.TOKEN_O]}.")

    def train_q_table(self):
        validation_interval = 1000
        episodes = 10000
        learning_rate = 0.001
        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                print(f"Testing after {episode} episodes")
                self.test(episodes=100)

            player_agents = [QTableAgent('X', self.q_table), QTableAgent('O', self.q_table)]
            game = Game()

            player_num = random.randint(0, 1)
            game.current_player_token = game.player_tokens[player_num]
            player_agent = player_agents[player_num]

            game_step = 0
            while not game.is_finished():
                observed_state = game.extract_qstate()
                state_value = player_agent.q_table[observed_state]

                game.make_move(player_agent)

                player_num = (player_num + 1) % 2
                game.current_player_token = game.player_tokens[player_num]
                player_agent = player_agents[player_num]

                next_observed_state = game.extract_qstate()
                if game.is_finished():
                    if game.winner_token == Game.TOKEN_X:
                        next_state_value = 1.0
                    elif game.winner_token == Game.TOKEN_O:
                        next_state_value = 0.0
                    else:
                        next_state_value = 0.5
                else:
                    next_state_value = player_agent.q_table[next_observed_state]

                player_agent.q_table[observed_state] += learning_rate * (next_state_value - state_value)

                game_step += 1

        if self.save_on_train:
            self.save()