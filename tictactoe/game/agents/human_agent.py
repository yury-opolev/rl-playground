
class HumanAgent(object):
    def __init__(self, player_token):
        self.player_token = player_token
        self.name = 'Human'

    def get_action(self, actions, game=None):
        if not actions:
            input("No moves for you...(hit enter)")
            return None

        while True:
            while True:
                action = input('"%s", please enter a move "<row>,<column>": ' % (self.player_token))
                action = self.try_parse_move(action)
                if action is None:
                    print('Bad format, enter e.g. "0,1"')
                else:
                    break

            if action in actions:
                return action
            else:
                print("You can't play that move")

        return None

    def try_parse_move(self, action):
        try:
            x, y = action.split(",")
            return (int(x), int(y))
        except:
            return None