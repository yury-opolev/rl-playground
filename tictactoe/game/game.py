import platform
import os
import random
import numpy as np

class Game:

    ROWS, COLS = (3, 3)
    EMPTYTOKEN = ' '
    TOKEN_X = 'x'
    TOKEN_O = 'o'
    TOKENS = [TOKEN_X, TOKEN_O]

    def clear_screen():
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    def __init__(self) -> None:
        self.grid = { (i,j) : Game.EMPTYTOKEN for i in range(Game.COLS) for j in range(Game.ROWS) }
        self.player_tokens = Game.TOKENS
        self.current_player_token = None
        self.winner_token = None

    def extract_features(self):
        features = []
        for cell in self.grid:
            cell_features = [0.0] * 2
            if cell == Game.TOKEN_X:
                cell_features[0] = 1.0
                cell_features[1] = 0.0
            elif cell == Game.EMPTYTOKEN:
                cell_features[0] = 0.0
                cell_features[1] = 0.0
            else:
                cell_features[0] = 0.0
                cell_features[1] = 1.0

            features += cell_features

        if Game.TOKEN_X == self.current_player_token:
            features += [1., 0.]
        else:
            features += [0., 1.]

        return np.array(features).reshape(1, -1)

    def play(self, player_agents, draw=False):
        player_num = random.randint(0, 1)
        self.current_player_token = self.player_tokens[player_num]
        while not self.is_finished():
            player_agent = player_agents[player_num]
            if draw:
                Game.clear_screen()
                print('Player "%s" is taking turn...' % (player_agent.player_token))
                self.draw()

            self.make_move(player_agent)
            player_num = (player_num + 1) % 2
            self.current_player_token = self.player_tokens[player_num]

        if draw:
            Game.clear_screen()
            print('Game is finished, player "%s" wins!' % (self.winner_token))
            self.draw()

        return self.winner_token

    def make_move(self, player_agent):
        actions = self.get_actions(player_agent.player_token)
        action = player_agent.get_action(actions, self) if actions else None

        if action:
            self.take_action(action, player_agent.player_token)

    def take_action(self, action, player_token):
        x, y = action
        self.grid[x,y] = player_token

    def undo_action(self, action, player_token):
        x, y = action
        self.grid[x,y] = Game.EMPTYTOKEN

    def get_actions(self, player_token):
        possible_actions = set()
        for xi in [0, 1, 2]:
            for yi in [0, 1, 2]:
                if (self.grid[xi,yi] == Game.EMPTYTOKEN):
                    possible_actions.add((xi, yi))

        return possible_actions

    def is_finished(self):
        self.winner_token = self.find_winner()
        occupied_cells = 0
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if (self.grid[i,0] != Game.EMPTYTOKEN):
                    occupied_cells += 1
        return (self.winner_token is not None) or (occupied_cells == 9)

    def find_winner(self):
        for token in self.player_tokens:
            for i in [0, 1, 2]:
                if (self.grid[i,0] == token) and (self.grid[i,1] == token) and (self.grid[i,2] == token):
                    return token
                if (self.grid[0,i] == token) and (self.grid[1,i] == token) and (self.grid[2,i] == token):
                    return token

            if (self.grid[0,0] == token) and (self.grid[1,1] == token) and (self.grid[2,2] == token):
                return token
            if (self.grid[2,0] == token) and (self.grid[1,1] == token) and (self.grid[0,2] == token):
                return token

        return None
    
    def get_opponent_token(self, given_token):
        for token in self.player_tokens:
            if token != given_token:
                return token

    def draw(self):
        print('  ┌───┬───┬───┐')
        print('0 │ ', end='')
        print(self.grid[0,0], end='')
        print(' │ ', end='')
        print(self.grid[0,1], end='')
        print(' │ ', end='')
        print(self.grid[0,2], end='')
        print(' │')

        print('  ├───┼───┼───┤')

        print('1 │ ', end='')
        print(self.grid[1,0], end='')
        print(' │ ', end='')
        print(self.grid[1,1], end='')
        print(' │ ', end='')
        print(self.grid[1,2], end='')
        print(' │')

        print('  ├───┼───┼───┤')

        print('2 │ ', end='')
        print(self.grid[2,0], end='')
        print(' │ ', end='')
        print(self.grid[2,1], end='')
        print(' │ ', end='')
        print(self.grid[2,2], end='')
        print(' │')

        print('  └───┴───┴───┘')
        print('    0   1   2  ')
