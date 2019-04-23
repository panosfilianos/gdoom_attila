class Agent():
    def __init__(self, s_size, a_size, trainer=None, player_mode=False):
        self.AC_network = AC_Network(s_size, a_size, self.name, trainer, player_mode)

    def get_policy_action(self, obs):
        with sess.as_default(), sess.graph.as_default():
            # Take an action using probabilities from policy network output.
            a_dist, v, rnn_state = sess.run([self.AC_network.policy, self.AC_network.value, self.AC_network.state_out],
                                            feed_dict={self.AC_network.inputs: [s],
                                                       self.AC_network.state_in[0]: rnn_state[0],
                                                       self.AC_network.state_in[1]: rnn_state[1]})

            action_index = self.choose_action_index(a_dist, deterministic=False)

            action = self.actions[action_index]

            return action, a_dist, v, rnn_state, action_index

    def get_custom_reward(self, env, game_reward):
        """
        Description
        --------------
        Final reward reshaped.

        Parameters
        --------------
        game_reward : float, reward provided by the environment
        """
        if params.scenario == 'basic':
            return game_reward / 100.0

        if params.scenario == 'defend_the_center':
            self.last_total_kills = self.env.game.get_game_variable(GameVariable.KILLCOUNT)
            return game_reward + self.get_ammo_reward() / 10

        if params.scenario == 'deadly_corridor':
            return (game_reward / 5 + self.get_health_reward() + self.get_kill_reward() + self.get_ammo_reward()) / 100.

        if params.scenario == 'my_way_home':
            return game_reward

        else:
            return game_reward