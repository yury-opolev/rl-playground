import os
import random

class Game:

    ROWS, COLS = (3, 3)
    EMPTYTOKEN = ' '
    TOKENS = ['x', 'o']

    def __init__(self) -> None:
        self.grid = { (i,j) : Game.EMPTYTOKEN for i in range(Game.COLS) for j in range(Game.ROWS) }
        self.player_tokens = Game.TOKENS
        self.winner_token = None

    def play(self, player_agents, draw=False):
        player_num = random.randint(0, 1)
        while not self.is_finished():
            player_agent = player_agents[player_num]
            if draw:
                os.system('clear')
                print('Player "%s" is taking turn...' % (player_agent.player_token))
                self.draw()

            self.make_move(player_agent, draw=draw)
            player_num = (player_num + 1) % 2

        if draw:
            os.system('clear')
            print('Game is finished, player "%s" wins!' % (self.winner_token))
            self.draw()

        return self.winner_token

    def make_move(self, player_agent, draw=False):
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
        return self.winner_token is not None

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
