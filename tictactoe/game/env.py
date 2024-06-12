import platform
import os
import random
import numpy as np

class Game:

    VALIDBOARDS_AXES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    ROWS, COLS = (3, 3)
    EMPTYTOKEN = ' '
    TOKEN_X = 'X'
    TOKEN_O = 'O'
    TOKENS = [TOKEN_X, TOKEN_O]

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.grid = { (i,j) : Game.EMPTYTOKEN for i in range(Game.COLS) for j in range(Game.ROWS) }
        self.player_tokens = Game.TOKENS
        self.current_player_token = None
        self.winner_token = None

        # return (reward, is_done)
        return None, False

    def extract_features(self):
        features = []
        for cellkey in self.grid:
            cell_features = [0.0] * 2
            if self.grid[cellkey] == Game.TOKEN_X:
                cell_features[0] = 1.0
                cell_features[1] = 0.0
            elif self.grid[cellkey] == Game.EMPTYTOKEN:
                cell_features[0] = 0.0
                cell_features[1] = 0.0
            else:
                cell_features[0] = 0.0
                cell_features[1] = 1.0
            features += cell_features

        if self.starting_player_token == Game.TOKEN_X:
            features += [1., 0.]
        else:
            features += [0., 1.]

        assert len(features) == 20, print("Should be 20 instead of {}".format(len(features)))
        return features

    def check_is_finished(self):
        # return (reward, is_done)

        if not self.is_finished():
            return 0.0, False

        if self.winner_token == Game.TOKEN_X:
            return 0.99, True

        if self.winner_token == Game.TOKEN_O:
            return -0.99, True

        return 0.0, True

    def starting_random_player(self):
        self.starting_player_token = np.random.choice([Game.TOKEN_X, Game.TOKEN_O])
        self.current_player_token = self.starting_player_token
        return self.starting_player_token

    def change_player(self):
        if self.current_player_token == Game.TOKEN_X:
            self.current_player_token = Game.TOKEN_O
            return self.current_player_token

        if self.current_player_token == Game.TOKEN_O:
            self.current_player_token = Game.TOKEN_X
            return self.current_player_token

    def step(self, action, grid, player_token):
        reward, done = self.check_is_finished()
        if done == True:
            return reward, done

        self.current_player_token = player_token
        grid[action] = player_token

        # return (reward, is_done)
        return self.check_is_finished()

    def take_action(self, action, player_token):
        x, y = action
        self.grid[x,y] = player_token

    def undo_action(self, action):
        x, y = action
        self.grid[x,y] = Game.EMPTYTOKEN

    def get_possible_actions(self):
        possible_actions = []
        for xi in [0, 1, 2]:
            for yi in [0, 1, 2]:
                if (self.grid[xi,yi] == Game.EMPTYTOKEN):
                    possible_actions += [(xi, yi)]

        return possible_actions

    def is_finished(self):
        self.winner_token = self.find_winner_token()
        if self.winner_token is not None:
            return True
        
        occupied_cells = 0
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if (self.grid[i,j] != Game.EMPTYTOKEN):
                    occupied_cells += 1
        return (occupied_cells == 9)

    def find_winner_token(self):
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
    
    def clear_screen(self):
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

    def draw(self):
        print( '  ┌───┬───┬───┐')
        print(f'0 │ {self.grid[0,0]} │ {self.grid[0,1]} │ {self.grid[0,2]} │')
        print( '  ├───┼───┼───┤')
        print(f'1 │ {self.grid[1,0]} │ {self.grid[1,1]} │ {self.grid[1,2]} │')
        print( '  ├───┼───┼───┤')
        print(f'2 │ {self.grid[2,0]} │ {self.grid[2,1]} │ {self.grid[2,2]} │')
        print( '  └───┴───┴───┘')
        print( '    0   1   2  ')
