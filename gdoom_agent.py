import gdoom_agent_utils
import numpy as np
import gdoom_ac
import os
import tensorflow as tf

import copy

from utils.network_params import *

from vizdoom import GameVariable

class Agent():
    def __init__(self, name, actions, s_size, a_size, trainer=None, player_mode=False):

        self.name = "agent_" + str(name)
        self.actions = actions
        gdoom_agent_utils.initialize_containers(self)
        self.summary_writer = tf.summary.FileWriter(params.summary_path + "/train_" + str(name))
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = gdoom_ac.AC_Network(s_size = s_size,
                                            a_size = a_size,
                                            scope = self.name,
                                            trainer=trainer)
        self.update_local_ops = gdoom_agent_utils.update_target_graph('global', self.name)

    def initialiaze_game_vars(self):
        """
        Description
        --------------
        Initialize game variables used for reward reshaping.

        """
        self.last_total_health = 100.0
        self.last_total_ammo2 = 52
        self.last_total_kills = 0

    def get_policy_action(self, sess, obs, rnn_state):
        with sess.as_default(), sess.graph.as_default():
            # Take an action using probabilities from policy network output.
            a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                            feed_dict={self.local_AC.inputs: [obs],
                                                       self.local_AC.state_in[0]: rnn_state[0],
                                                       self.local_AC.state_in[1]: rnn_state[1]})

            #deterministic was False on the original code, change to False to debug
            action_index = gdoom_agent_utils.choose_action_index(policy=a_dist,
                                                                 deterministic=False)

            return a_dist, v, rnn_state, action_index

    def get_custom_reward(self, env, game_reward):
        """
        Description
        --------------
        Final reward reshaped.

        Parameters
        --------------
        game_reward : float, reward provided by the environment
        """
        env.agent.last_total_health = env.game.get_game_variable(GameVariable.HEALTH)
        env.agent.last_total_ammo2 = env.game.get_game_variable(GameVariable.AMMO2)
        env.agent.last_total_kills = env.game.get_game_variable(GameVariable.KILLCOUNT)
        return game_reward

        if params.scenario == 'basic':
            return game_reward / 100.0

        if params.scenario == 'defend_the_center':
            # self.last_total_kills = env.game.get_game_variable(GameVariable.KILLCOUNT)
            return game_reward + gdoom_agent_utils.get_kill_reward(env=env) #+ gdoom_agent_utils.get_ammo_reward(env=env) / 10

        if params.scenario == 'deadly_corridor':
            return (game_reward / 5 + gdoom_agent_utils.get_health_reward(env=env) + gdoom_agent_utils.get_kill_reward(env=env) + gdoom_agent_utils.get_ammo_reward(env=env)) / 100.

        if params.scenario == 'my_way_home':
            return game_reward

        else:
            return game_reward

    def retrain_handle(self, sess, s, ep_done, max_episodes, gamma, rnn_state):
        with sess.as_default(), sess.graph.as_default():
            if (len(self.local_AC.episode_buffer) == params.n_steps and
                ep_done != True and
                self.episode_step_count != max_episodes - 1):
                # print("RETRAINING WEIGHTS ON THE NETWORK")
                # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                v1 = sess.run(self.local_AC.value,
                              feed_dict={self.local_AC.inputs: [s],
                                         self.local_AC.state_in[0]: rnn_state[0],
                                         self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                Losses_grads = self.retrain_weights(self.local_AC.episode_buffer, sess, gamma, v1)

                # if params.use_curiosity:
                #     self.v_l, self.p_l, self.e_l, self.Inv_l, self.Forward_l, self.g_n, self.v_n = Losses_grads
                # else:
                self.v_l, self.p_l, self.e_l, self.g_n, self.v_n = Losses_grads

                # Empty buffer
                self.local_AC.episode_buffer = []

                # Copy the global network weights to the local network
                sess.run(self.update_local_ops)
                # if params.use_curiosity: sess.run(self.update_local_ops_P)


    def retrain_weights(self,rollout,sess,gamma,bootstrap_value):
        """
        Description
        --------------
        Unroll trajectories to train the model.

        Parameters
        --------------
        rollout            : list, buffer containing experiences.
        sess               : Tensorflow session
        gamma              : Float, discount factor
        bootstrap_value    : Float, bootstraped value function if episode is not finished
        """
        rollout = np.array(rollout)
        observations, actions, rewards, next_observations, _, values = rollout.T

        # Process the rollout by constructing variables for the loss functions
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = gdoom_agent_utils.discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = discounted_rewards - self.value_plus[:-1]

        # Update the local Actor-Critic network using gradients from loss
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     self.local_AC.inputs:np.vstack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages,
                     self.local_AC.state_in[0]:self.batch_rnn_state[0],
                     self.local_AC.state_in[1]:self.batch_rnn_state[1]}

        if params.use_ppo:
            old_policy = sess.run([self.local_AC.responsible_outputs],feed_dict=feed_dict)
            feed_dict.update({self.local_AC.old_policy:old_policy[0]})

        self.v_l,self.p_l,self.e_l,self.g_n,self.v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
                                                                                         self.local_AC.policy_loss,
                                                                                         self.local_AC.entropy,
                                                                                         self.local_AC.grad_norms,
                                                                                         self.local_AC.var_norms,
                                                                                         self.local_AC.state_out,
                                                                                         self.local_AC.apply_grads],
                                                                                        feed_dict=feed_dict)

        Losses = [self.v_l, self.p_l, self.e_l]
        Grad_vars = [self.g_n, self.v_n]

        # Update the local ICM network using gradients from loss
        if params.use_curiosity:
            feed_dict_P = {self.local_Pred.s1:np.vstack(observations),
                           self.local_Pred.s2:np.vstack(next_observations),
                           self.local_Pred.aindex:actions}

            self.Inv_l, self.Forward_l, _ = sess.run([self.local_Pred.invloss,
                                                      self.local_Pred.forwardloss,
                                                      self.local_Pred.apply_grads],
                                                     feed_dict=feed_dict_P)

            Losses += [self.Inv_l,self.Forward_l]

        return list(np.array(Losses)/len(rollout))+Grad_vars

    def after_ep_util_handle(self, sess, gamma, saver, model_path):
        # Update containers for tensorboard summary
        gdoom_agent_utils.update_containers(self)

        # Update the network using the episode buffer at the end of the episode.
        if len(self.local_AC.episode_buffer) != 0:
            if params.use_curiosity:
                self.v_l, self.p_l, self.e_l, self.Inv_l, self.Forward_l, self.g_n, self.v_n = self.retrain_weights(self.local_AC.episode_buffer,
                                                                                                          sess,
                                                                                                          gamma,
                                                                                                          0.0)
            else:
                self.v_l, self.p_l, self.e_l, self.g_n, self.v_n = self.retrain_weights(self.local_AC.episode_buffer,
                                                                              sess,
                                                                              gamma,
                                                                              0.0)

        # Periodically save gifs of episodes
        if self.name == 'agent_0' and self.episode_count % params.freq_gif_save == 0 and self.episode_count != 0:
            time_per_step = 0.05
            images = np.array(self.episode_frames)
            gif_path = os.path.join(params.frames_path, 'gyma3c' + str(self.episode_count) + '.gif')
            gdoom_agent_utils.make_gif(images, gif_path)

        # Periodically save model parameters
        if self.episode_count % params.freq_model_save == 0 and self.name == 'agent_0' and self.episode_count != 0:
            saver.save(sess, model_path + '/model-' + str(self.episode_count) + '.cptk')
            print("Saved Model")

        # Periodically save summary statistics
        if self.episode_count % params.freq_summary == 0 and self.episode_count != 0:
            gdoom_agent_utils.update_summary(self)

        self.episode_count += 1