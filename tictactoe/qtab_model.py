import random
import pickle
import numpy as np

import copy

from pathlib import Path

from game.env import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent

class QTabModel(object):
    def __init__(self):
        self.q_table = {}

        self.learning_rate = 0.001 #0.001
        self.gamma = 1.0

    def get_action_index(self, action):
        x, y = action
        action_index = x * 3 + y
        return action_index
    
    def get_state_action_value(self, state, action):
        state_key = self.get_state_key(state)
        if not state_key in self.q_table:
            self.q_table[state_key] = np.zeros(9)
        action_index = self.get_action_index(action)
        return self.q_table[state_key][action_index]
    
    def get_state_key(self, state):
        result = ''
        for item_1,item_2 in zip(state[0::2], state[1::2]):
            if item_1 == 1:
                result += 'X'
            elif item_2 == 1:
                result += 'O'
            else:
                result += '-'
        return result

    def get_output(self, state, action):
        return self.get_state_action_value(state, action)

    def test(self, episodes=1000, print_failed_games=False):
        ai_wins = { 'WON': 0, 'LOST': 0, 'DRAW': 0 }
        for episode in range(episodes):
            game = Game()

            if random.choice([0, 1]) == 0:
                player_agents = [AIAgent('X', self), RandomAgent('O')]
                ai_agent_token = Game.TOKEN_X
                random_agent_token = Game.TOKEN_O
            else:
                player_agents = [RandomAgent('X'), AIAgent('O', self)]
                ai_agent_token = Game.TOKEN_O
                random_agent_token = Game.TOKEN_X
            
            game.starting_player_token = Game.TOKEN_X
            game.current_player_token = game.starting_player_token

            current_player_agent = self.get_player_agent(game, player_agents)

            game_history = []
            while not game.is_finished():
                observed_state = game.extract_features()
                actions = game.get_possible_actions()
                action, value = current_player_agent.get_action(actions, game)
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                next_observed_state = game.extract_features()
                game_history.append((self.get_state_key(observed_state), action, self.get_state_key(next_observed_state)))

            winner_token = game.winner_token
            if winner_token is None or winner_token == Game.EMPTYTOKEN:
                ai_wins['DRAW'] = ai_wins['DRAW'] + 1
            else:
                if ai_agent_token == winner_token:
                    ai_wins['WON'] = ai_wins['WON'] + 1
                else:
                    ai_wins['LOST'] = ai_wins['LOST'] + 1

            if print_failed_games and winner_token == random_agent_token:
                print(game.get_string())

        print(f"AI draws: {ai_wins['DRAW']}, wins: {ai_wins['WON']}, losses: {ai_wins['LOST']}.")
        return (ai_wins['DRAW'], ai_wins['WON'], ai_wins['LOST'])

    def train(self, episodes=10000, epsilon=0.5, validate=False):
        validation_interval = 10000
        for episode in range(episodes):
            if validate and (episode % validation_interval == 0):
                print(f"Testing after {episode} episodes:")
                self.test()
                print()

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            current_player_agent = self.get_player_agent(game, player_agents)

            is_done = False
            single_game_history = []
            while not is_done:
                # get state S 
                observed_state = game.extract_features()

                # get action A (and Q(S, A)) 
                actions = game.get_possible_actions()
                (action, action_value) = current_player_agent.get_action(actions, game, epsilon)

                # get R
                reward, is_done = game.step(action, game.grid, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                # get S'
                next_observed_state = game.extract_features()

                # get action A' (and Q(S',A'))
                if is_done:
                    best_next_state_action_value = 0.0
                else:
                    actions_next = game.get_possible_actions()
                    (best_next_action, best_next_state_action_value) = current_player_agent.get_action(actions_next, game, 0.0)

                single_game_history.append((observed_state, action, reward, best_next_state_action_value))

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                self.update_weights(observed_state, action, reward, best_next_state_action_value)

        print(f"Final testing:")
        (draw, win_x, win_o) = self.test()
        print()

        return (draw, win_x, win_o)

    def update_weights(self, state, action, reward, next_state_action_value):
        # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
        state_key = self.get_state_key(state)
        action_index = self.get_action_index(action)

        state_action_value = self.q_table[state_key][action_index]
        self.q_table[state_key][action_index] = state_action_value + self.learning_rate * (reward + self.gamma * next_state_action_value - state_action_value)

    def restore_weights(self, path):
        existing_modelfile = Path(path)
        if not existing_modelfile.exists():
            return
        
        print(f"Loading from {path}")
        print()
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

    def save_weights(self, path):
        print(f"Saving to {path}")
        print()
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def get_player_agent(self, game, player_agents):
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]
        return current_player_agent
