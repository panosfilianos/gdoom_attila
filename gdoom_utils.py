def setup_network(sess, AC_network, agent):
    with sess.as_default(), sess.graph.as_default():
        # Copy the global networks weights to local network weights
        sess.run(self.update_local_ops)
        if params.use_curiosity: sess.run(self.update_local_ops_P)

        # Initialize buffer for training
        AC_network.episode_buffer = []

        # Initialize frames buffer to save gifs
        agent.episode_frames = []

        # Initialize variables to record performance for tensorflow summary
        agent.episode_values = []
        agent.episode_reward = 0
        agent.episode_curiosity = 0
        agent.episode_step_count = 0

        # Initialize game vars (health, kills ...)
        agent.initialiaze_game_vars()

        # Begin new episode
        d = False
        agent.game.new_episode()
        agent.episode_st = time.time()
        #
        # # Initialize LSTM gates
        rnn_state = agent.local_AC.state_init
        agent.batch_rnn_state = rnn_state
        #
        # Get first state and process it
        s = agent.get_game.get_state().screen_buffer
        agent.episode_frames.append(s)
        s = process_frame(s, crop, resize)
        #
        # while self.env.is_episode_finished() == False:
        #
        #     # Take an action using probabilities from policy network output.
        #     a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
        #                                     feed_dict={self.local_AC.inputs: [s],
        #                                                self.local_AC.state_in[0]: rnn_state[0],
        #                                                self.local_AC.state_in[1]: rnn_state[1]})
        #
        #     action_index = self.choose_action_index(a_dist, deterministic=False)
        #
        #     # Get extrinsic reward
        #     if params.no_reward:
        #         reward = 0
        #     else:
        #         reward = self.get_custom_reward(self.env.make_action(self.actions[action_index], 2))
        #
        #     # Check if episode is finished to process next state
        #     done = self.env.is_episode_finished()
        #     if done == False:
        #         s1 = self.env.get_state().screen_buffer
        #         episode_frames.append(s1)
        #         s1 = process_frame(s1, crop, resize)
        #     else:
        #         s1 = s
        #
        #     # Get intrinsic reward
        #     if params.use_curiosity:
        #         curiosity = np.clip(self.local_Pred.pred_bonus(s, s1, a_dist[0]), -1, 1) / 5
        #         self.episode_curiosity += curiosity
        #     else:
        #         curiosity = 0
        #
        #     # Total reward
        #     r = curiosity + reward
        #
        #     # Append step to buffer
        #     episode_buffer.append([s, action_index, r, s1, d, v[0, 0]])
        #
        #     # Update variables
        #     self.episode_values.append(v[0, 0])
        #     self.episode_reward += r
        #     s = s1
        #     total_steps += 1
        #     self.episode_step_count += 1
        #
        #     # If the episode hasn't ended, but maximum steps is reached, we update the global network using the current rollout.
        #     if len(
        #             episode_buffer) == params.n_steps and done != True and self.episode_step_count != max_episodes - 1:
        #         # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
        #         v1 = sess.run(self.local_AC.value,
        #                       feed_dict={self.local_AC.inputs: [s],
        #                                  self.local_AC.state_in[0]: rnn_state[0],
        #                                  self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
        #
        #         Losses_grads = self.train(episode_buffer, sess, gamma, v1)
        #
        #         if params.use_curiosity:
        #             self.v_l, self.p_l, self.e_l, self.Inv_l, self.Forward_l, self.g_n, self.v_n = Losses_grads
        #         else:
        #             self.v_l, self.p_l, self.e_l, self.g_n, self.v_n = Losses_grads
        #
        #         # Empty buffer
        #         episode_buffer = []
        #
        #         # Copy the global network weights to the local network
        #         sess.run(self.update_local_ops)
        #         if params.use_curiosity: sess.run(self.update_local_ops_P)
        #
        #     if done == True:
        #         # Print perfs of episode
        #         self.print_end_episode_perfs()
        #         break
        #
        # # Update containers for tensorboard summary
        # self.update_containers()
        #
        # # Update the network using the episode buffer at the end of the episode.
        # if len(episode_buffer) != 0:
        #     if params.use_curiosity:
        #         self.v_l, self.p_l, self.e_l, self.Inv_l, self.Forward_l, self.g_n, self.v_n = self.train(
        #             episode_buffer, sess, gamma, 0.0)
        #     else:
        #         self.v_l, self.p_l, self.e_l, self.g_n, self.v_n = self.train(episode_buffer, sess, gamma, 0.0)
        #
        #         # Periodically save gifs of episodes
        # if self.name == 'worker_0' and self.episode_count % params.freq_gif_save == 0 and self.episode_count != 0:
        #     time_per_step = 0.05
        #     images = np.array(episode_frames)
        #     gif_path = os.path.join(params.frames_path, 'image' + str(self.episode_count) + '.gif')
        #     make_gif(images, gif_path)
        #
        # # Periodically save model parameters
        # if self.episode_count % params.freq_model_save == 0 and self.name == 'worker_0' and self.episode_count != 0:
        #     saver.save(sess, self.model_path + '/model-' + str(self.episode_count) + '.cptk')
        #     print("Saved Model")
        #
        # # Periodically save summary statistics
        # if self.episode_count % params.freq_summary == 0 and self.episode_count != 0:
        #     self.update_summary()
        #
        # self.episode_count += 1
        #
        # print("{} finished {} episodes with {} steps. Going to sleep ..".format(self.name, self.episode_count,
        #                                                                         total_steps))
