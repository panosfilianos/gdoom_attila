import logging
import os
import pygame

import copy

import numpy as np

import gym
from gym import spaces, error
import vizdoom

import gdoom_agent
import gdoom_utils
import gdoom_agent_utils

from utils.network_params import *

from train import train_agents

CPU = 101
HUMAN = 102

logger = logging.getLogger(__name__)

# Constants
NUM_ACTIONS = 43
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

from vizdoom import Button

keymap = {Button.MOVE_LEFT: ( ord('a'), "strafe left"),
          Button.MOVE_RIGHT: (ord('d'), "strafe right"),
          Button.TURN_LEFT: (276, "turn left"),
          Button.TURN_RIGHT: (275, "turn right"),
          Button.ATTACK: (ord(' '), "shoot the fool"),
          Button.MOVE_FORWARD: (ord('w'), "move forward"),
          Button.MOVE_BACKWARD: (ord('s'), "move backward"),
         }

name_to_settings_index_dict = {
    'basic': 0,
    'deadly_corridor': 1,
    'defend_the_center': 2,
    'defend_the_line': 3,
    'health_gathering': 4,
    'my_way_home': 5,
    'predict_position': 6,
    'take_cover': 7,
    'deathmatch': 8,
}

DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                               # 0  - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],  # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],              # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                  # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],         # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],      # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                          # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, [x for x in range(NUM_ACTIONS) if x != 33], 0, 20] # 8 - Deathmatch
]

from vizdoom import GameVariable
collect_variables = [ ('kills', GameVariable.KILLCOUNT),
                      ('ammo', GameVariable.AMMO2),
                      ('health', GameVariable.HEALTH)]




class GDoomEnv(gym.Env):
    #added parameters needed to initialize the agent/worker

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, i, state_size, action_size, trainer, model_path, level=0, enable_sound=False):
        # tue.
        self.name = i
        self.enable_sound = enable_sound
        self.is_initialized = False
        self.screen_height = 240
        self.screen_width = 320
        self.model_path = model_path

        # here set how many frames you save: this will affect the size of the data later on
        # in the NN -- chnaged to 1 from 3
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.game = vizdoom.DoomGame()
        self.game.load_config(os.path.join("scenarios", params.scenario + ".cfg"))
        self.game.set_doom_scenario_path(os.path.join("scenarios", params.scenario + ".wad"))
        self.game.set_window_visible(False)
        self.game.init()

        self.accumulated_reward = 0

        self.actions = gdoom_utils.button_combinations(scenario=params.scenario)
        #actions
        # [move left, move right, shoot, move forward, move backwards, turn left, turn right]

        self.agent = gdoom_agent.Agent(actions = self.actions,
                                       s_size = state_size,
                                       a_size = action_size,
                                       trainer=trainer,
                                       name = self.name)
        self.curr_seed = 0
        self.mode = CPU # or human
        #from A3C name of level to GDoom settings index
        self.level = name_to_settings_index_dict[level]
        self.reset() # load buttons, etc.




    def get_keys_to_action(self):
        abut = self.game.get_available_buttons()
        print(abut)
        k2a = {}
        for k,code in enumerate(abut):
            if code not in keymap:
                raise Exception("Letter not defined in keymap")
            d =keymap[code]
            letter = d[0]
            if self.mode != HUMAN:
                print("%i: '%s' -> %s"%(k, chr(letter), d[1]))
            k2a[letter] = k
        k2a = { (k,): a for k, a in k2a.items()}
        return k2a


    def _configure(self, lock=None, **kwargs):
        if 'screen_resolution' in kwargs:
            logger.warn('Deprecated - Screen resolution must now be set using a wrapper. See documentation for details.')
        # Multiprocessing lock
        if lock is not None:
            self.lock = lock

    def th_start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()


    def th_load_level(self):
        # Closing if is_initialized
        if self.is_initialized:
            self.is_initialized = False
            self.game.close()
            self.game = vizdoom.DoomGame()

        dp = os.path.dirname(vizdoom.__file__)
        self.game = vizdoom.DoomGame()

        # self.level = 0

        scenario = dp + "/scenarios/" + DOOM_SETTINGS[self.level][CONFIG]
        print("Doom> Loading level: " + scenario)
        self.game.load_config(scenario)
        self.game.set_sound_enabled(self.enable_sound)
        # self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        self.game.set_window_visible(False)
        self.game.get_available_buttons_size()

        self.game.set_available_game_variables([])
        for _, gv in collect_variables:
            self.game.add_available_game_variable(gv)

        self.game.init()
        abut = self.game.get_available_buttons()

        self.action_space = gym.spaces.Discrete(len(abut))

        self.is_initialized = True
        self._closed = False
        self.th_start_episode()


    def render(self, mode="human"):
        raise NotImplementedError("Please implement a render function to make movies, etc.")
        if mode == 'rgb_array':
            return np.array(...)  # return RGB frame suitable for video
        elif mode is 'human':
            pass
        else:
            super().render(mode=mode)  # just raise an exception
        pass

    def step(self, action_index):
        if self.mode == CPU:
            skiprate = 4
        else:
            skiprate = 1 ##for human

        # prev_misc = self.game.get_state().game_variables
        # misc = prev_misc
        action_index = int(action_index)
        assert(isinstance(action_index, int))
        if action_index >= 0:

            a_t = self.actions[action_index]
            reward = self.agent.get_custom_reward(self, self.game.make_action(a_t, skiprate))
        elif action_index == -1:
            # get reward from executing action
            reward = self.agent.get_custom_reward(self, self.game.make_action(  np.zeros([self.action_space.n]).tolist()   ), skiprate)##this and following if are equal..no sense to differentiate the mode
        elif self.mode == CPU:
            raise Exception("Error")

        #added from a3c
        # r_t = self.agent.get_custom_reward(self.env.make_action(self.actions[action_index], 2))


        # r_t = self.game.get_last_reward()#???????????????????????????????/
        is_finished = self.game.is_episode_finished()

        if is_finished:
            if self.mode == HUMAN or self.mode == CPU:
                # print("Total reward accumulated: ", self.info['accumulated_reward'], " time alive ", self.info['time_alive'], " kills ", self.info['kills'])
                pass
            info = {}
            # image_buffer = np.zeros(shape=(self.screen_height, self.screen_width), dtype=np.uint8)
            image_buffer = None

        else:
            state = self.game.get_state()
            image_buffer = state.screen_buffer
            self.agent.episode_frames.append(image_buffer)

            # misc = state.game_variables
            # info = {s: misc[k] for k,(s,_) in enumerate(collect_variables)}##it just saves the variables in a dictionary
            #
            # self.info['time_alive'] += 1
            # self.info['kills'] = info['kills'] # total kills
            # if False: ##this is never executed
            #     import matplotlib.pyplot as plt
            #     plt.imshow(image_buffer.copy().swapaxes(0,1))
            #     plt.show()

        # it is shaped before sent out
        # reward = self.shape_reward(reward, misc, prev_misc) ##reward of the last game

        # self.info['accumulated_reward'] += reward
        #
        # info = self.info.copy()
        # info['accumulated_reward'] = self.accumulated_reward
        # info['reward'] = reward
        # info['time_alive'] = self.time_alive
        return image_buffer, reward, is_finished#, info

    def get_HWC(self):
        return 0

    def shape_reward(self, r_t, misc, prev_misc, t=None): ##changed this one in prev_misc
        # CHANGED THIS to be a multiplication of the difference between them
        # Check any kill countprev_misc
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]):  # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]):  # Loss HEALTH
            r_t = r_t - 0.1

        return r_t


    def reset(self):
        if self.is_initialized and not self._closed:
            self.th_start_episode()
        else:
            self.th_load_level()
        image_buffer = self.game.get_state().screen_buffer
        self.accumulated_reward = 0
        self.time_alive = 0
        self.info = {}
        self.info['time_alive'] = 0
        self.info['accumulated_reward'] = 0
        self.info['kills'] = 0
        return image_buffer.copy()

def gdoom_openaigym_wrapper(Cls):
    class NewCls(object):
        def __init__(self,level=2, frame_size=64, mode=CPU, *args,**kwargs):
            self.env = Cls(level=level, *args, **kwargs)

        def __getattribute__(self,s):
            """
            this is called whenever any attribute of a NewCls object is accessed. This function first tries to
            get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
            instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and
            the attribute is an instance method then `time_this` is applied.
            """
            try:
                x = super(NewCls,self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x

            try:
                x = self.env.__getattribute__(s)
            except:
                # FrameStack > GPreprocessFrame > GRewardScaler > WGDoomEnv
                x = self.env.env.env.env.__getattribute__(s)
            if type(x) == type(self.__init__): # it is an instance method
                return x                 # this is equivalent of just decorating the method with time_this
            else:
                return x
    return NewCls

def train(env, max_episodes, gamma, sess, coord, saver, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    #added parameters needed from train.py

    pressed_keys = []
    running = True
    env_done = True

    total_frames = 0
    clock = pygame.time.Clock()

    env.agent.episode_count = 0
    with sess.as_default(), sess.graph.as_default():

        # while running:
        while (not coord.should_stop()) and (env.agent.episode_count < max_episodes):

            rnn_state = gdoom_utils.setup_network(sess, env.agent)
            env.game.new_episode()
            #reset the frames to gif every episode
            ep_done = False
            # obs = env.reset()

            #reset loss arrays (do it here, because we need to initialize it first time and make sure it's not empty printed)
            env.agent.v_l_array = []
            env.agent.p_l_array = []
            env.agent.e_l_array = []
            env.agent.t_l_array = []


            obs = env.game.get_state().screen_buffer
            env.agent.episode_frames.append(obs)
            obs = gdoom_utils.process_frame(obs, crop, resize)


            while(not(ep_done)):
                a_dist, v, rnn_state, action_index = env.agent.get_policy_action(sess=sess,
                                                                                obs=obs,
                                                                                rnn_state=rnn_state)

                prev_obs = obs
                # obs, rew, ep_done, info = env.step(action_index)
                obs, rew, ep_done = env.step(action_index)
                if (ep_done == True):
                    obs = prev_obs
                else:
                    obs = gdoom_utils.process_frame(obs, crop, resize)

                if callback is not None:
                    #train the network here

                    if (callback(env=env,
                                 prev_obs=prev_obs,
                                 obs=obs,
                                 action=action_index,
                                 reward=rew,
                                 ep_done=ep_done,
                                 v=v,
                                 sess=sess,
                                 max_episodes=max_episodes,
                                 gamma=gamma,
                                 rnn_state=rnn_state)):
                        break

            env.agent.after_ep_util_handle(sess=sess,
                                           gamma=gamma,
                                           saver=saver,
                                           model_path =env.model_path)

            total_frames += env.agent.episode_step_count
            # reset all episode stats

            env.agent.episode_values = []
            env.agent.episode_reward = 0
            env.agent.episode_step_count = 0

            # pygame.display.flip()
            clock.tick(fps)
        pygame.quit()


def after_step_callback(env, prev_obs, obs, action, reward, ep_done, v, sess, max_episodes, gamma, rnn_state):
    # Append step to buffer
    env.agent.local_AC.episode_buffer.append([prev_obs, action, reward, obs, False, v[0, 0]])
    #consider
    # env.agent.local_AC.episode_buffer.append([prev_obs, action, reward, obs, ep_done, v[0, 0]])

    # Update variables
    env.agent.episode_values.append(v[0, 0])
    env.agent.episode_reward += reward
    env.agent.episode_step_count += 1

    env.agent.retrain_handle(sess=sess,
                         s=obs,
                         ep_done=ep_done,
                         max_episodes=max_episodes,
                         gamma=gamma,
                         rnn_state=rnn_state)

    if ep_done == True:
        # Print perfs of episode
        if (env.agent.name == "agent_0"):
            gdoom_agent_utils.print_end_episode_perfs(agent=env.agent)
        return True





def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    pass

@gdoom_openaigym_wrapper
class WGDoomEnv(GDoomEnv):
    pass

def getPossibleAction(scenario):
    if scenario == "basic" or scenario == "defend_the_center":
        return np.identity(3, dtype=int).tolist()
    else:
        return np.identity(6,dtype=int).tolist()

if __name__ == "__main__":
    print("GDoomEnv called")

    import numpy as np

    # Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
    genv = WGDoomEnv(level=2, frame_size=89)
    a, _, _, _ = genv.step(0)
    print( np.asarray(a).shape )

    # Also make a GPU environment, but using openai:
    env_cpu = gym.make("doom_scenario2_96-v0")
    frame = env_cpu.reset()
    print("Frame size for cpu player: ", np.asarray(frame).shape)

    env_human = gym.make("doom_scenario2_human-v0")
    frame = env_human.reset()
    print("Frame size for human player: ", np.asarray(frame).shape)