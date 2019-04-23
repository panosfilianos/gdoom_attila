class Agent():
    def __init__(self, s_size, a_size, trainer=None, player_mode=False):
        self.AC_network = AC_Network(s_size, a_size, self.name, trainer, player_mode)
