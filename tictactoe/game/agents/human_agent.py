
class HumanAgent(object):
    def __init__(self, player_token):
        self.player_token = player_token
        self.name = 'Human'

    def get_action(self, moves, game=None):
        if not moves:
            input("No moves for you...(hit enter)")
            return None

        while True:
            while True:
                move = input('"%s", please enter a move "<row>,<column>": ' % (self.player_token))
                move = self.try_parse_move(move)
                if move is None:
                    print('Bad format, enter e.g. "0,1"')
                else:
                    break

            if move in moves:
                return move
            else:
                print("You can't play that move")

        return None

    def try_parse_move(self, move):
        try:
            x, y = move.split(",")
            return (int(x), int(y))
        except:
            return None