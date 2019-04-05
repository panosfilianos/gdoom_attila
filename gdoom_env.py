import logging
import os

import numpy as np

import gym
from gym import spaces, error
import vizdoom

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

# Format (config, scenario, map, difficulty, actions, min, target)
# keymap = {0: (' ', "Shoot"), # shoot
#          14: ('a', "Turn left"), # turn left,
#          15: ('d', "Turn right"),  # turn right,
#          10: ('z', "strafe left"),  # strafe left,
#          11: ('x', "strafe right"),  # strafe right,
#          13: ('w', "Move forward"), # move forward
#          16: ('s', "Move back"),  # move backward
#         }

from vizdoom import Button

keymap = {Button.MOVE_LEFT: ( ord('a'), "strafe left"),
          Button.MOVE_RIGHT: (ord('d'), "strafe right"),
          Button.TURN_LEFT: (276, "turn left"),
          Button.TURN_RIGHT: (275, "turn right"),
          Button.ATTACK: (ord(' '), "shoot the fool"),
          Button.MOVE_FORWARD: (ord('w'), "move forward"),
          Button.MOVE_BACKWARD: (ord('s'), "move backward"),
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
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level=0, enable_sound=False):
        # tue.
        self.enable_sound = enable_sound
        self.is_initialized = False
        self.screen_height = 480
        self.screen_width = 640

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.game = vizdoom.DoomGame()
        self.accumulated_reward = 0

        self.curr_seed = 0
        self.mode = CPU # or human
        self.level = level
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
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
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

    def step(self, action):
        if self.mode == CPU:
            skiprate = 4
        else:
            skiprate = 1 ##for human

        prev_misc = self.game.get_state().game_variables
        misc = prev_misc
        assert(isinstance(action, int))
        if action >= 0:
            a_t = np.zeros([self.action_space.n])
            a_t[action] = 1
            a_t = a_t.astype(int)
            self.game.make_action(a_t.tolist())
        elif action == -1 and self.mode == HUMAN:
            self.game.make_action(  np.zeros([self.action_space.n]).tolist()   )##this and following if are equal..no sense to differentiate the mode
        elif action == -1 and self.mode == CPU:
            self.game.make_action(  np.zeros([self.action_space.n]).tolist()   )
        elif self.mode == CPU:
            raise Exception("Error")

        self.game.advance_action(skiprate)
        r_t = self.game.get_last_reward()#???????????????????????????????/
        is_finished = self.game.is_episode_finished()
        state = self.game.get_state()
        if is_finished:
            if self.mode == HUMAN or self.mode == CPU:
                print("Total reward accumulated: ", self.info['accumulated_reward'], " time alive ", self.info['time_alive'], " kills ", self.info['kills'])
            info = {}
            image_buffer = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        else:
            image_buffer = state.screen_buffer
            image_buffer = np.transpose(image_buffer.copy(), [1, 2, 0])
            misc = state.game_variables
            info = {s: misc[k] for k,(s,_) in enumerate(collect_variables)}##it just saves the variables in a dictionary

            self.info['time_alive'] += 1
            self.info['kills'] = info['kills'] # total kills
            if False: ##this is never executed
                import matplotlib.pyplot as plt
                plt.imshow(image_buffer.copy().swapaxes(0,1))
                plt.show()

        r_t = self.shape_reward(r_t, misc, prev_misc) ##reward of the last game

        self.info['accumulated_reward'] += r_t

        info = self.info.copy()
        # info['accumulated_reward'] = self.accumulated_reward
        # info['reward'] = r_t
        # info['time_alive'] = self.time_alive
        return image_buffer, r_t, is_finished, info

    def get_HWC(self):
        return 0
    def shape_reward(self, r_t, misc, prev_misc, t=None): ##changed this one in prev_misc
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

        # image_buffer = np.transpose(image_buffer, [2, 1, 0])
        return np.transpose(image_buffer.copy(), [1, 2, 0])


from gym.envs.gdoom.wrappers.gdoom_wrappers import SetPlayingMode
from gym.envs.gdoom.wrappers.gdoom_wrappers import GRewardScaler, GPreprocessFrame
from baselines.common.atari_wrappers import FrameStack

def gdoom_openaigym_wrapper(Cls):
    class NewCls(object):
        def __init__(self,level=2, frame_size=64, mode=CPU, *args,**kwargs):
            self.env = Cls(level=level, *args, **kwargs)
            self.env = GRewardScaler(self.env, scale=1)
            if mode == CPU:
                self.env = GPreprocessFrame(self.env, size=frame_size)
                self.env = FrameStack(self.env, 3)
            else:
                self.env = SetPlayingMode(target_mode=HUMAN)(self.env)

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
            x = self.env.__getattribute__(s)
            if type(x) == type(self.__init__): # it is an instance method
                return x                 # this is equivalent of just decorating the method with time_this
            else:
                return x
    return NewCls



def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    
    train_agents()

    # obs_s = env.observation_space
    # assert type(obs_s) == gym.spaces.box.Box
    # assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])
    #
    # if keys_to_action is None:
    #     if hasattr(env, 'get_keys_to_action'):
    #         keys_to_action = env.get_keys_to_action()
    #     elif hasattr(env.unwrapped, 'get_keys_to_action'):
    #         keys_to_action = env.unwrapped.get_keys_to_action()
    #     else:
    #         assert False, env.spec.id + " does not have explicit key to action mapping, " + \
    #                       "please specify one manually"
    # relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))
    #
    # if transpose:
    #     video_size = env.observation_space.shape[1], env.observation_space.shape[0]
    # else:
    #     video_size = env.observation_space.shape[0], env.observation_space.shape[1]
    #
    # if zoom is not None:
    #     video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)
    #
    # pressed_keys = []
    # running = True
    # env_done = True
    #
    # screen = pygame.display.set_mode(video_size)
    # clock = pygame.time.Clock()
    #
    #
    # while running:
    #
    #     if env_done:
    #         env_done = False
    #         obs = env.reset()
    #     else:
    #
    #         #THIS IS WHERE IT GETS ITS ACTIONS
    #         #THE ACTION ITSELF IS THE VALUE OF THE DICT:
    #         #print(keys_to_action.get(tuple(sorted(pressed_keys)), 0))
    #         #THE DICT HAS KEY AS ASCII VALUE OF THE BUTTON TO PRESS AND VALUE AS THE VALUE GIVEN TO IT BY THE ENV
    #         action = keys_to_action.get(tuple(sorted(pressed_keys)), 1)
    #         #added here because if you scroll those events in the pygame loop are registered only after human input
    #         # pressed_keys.append(action)
    #
    #         prev_obs = obs
    #         obs, rew, env_done, info = env.step(action)
    #         if callback is not None:
    #             callback(prev_obs, obs, action, rew, env_done, info)
    #
    #
    #     pygame.display.flip()
    #     clock.tick(fps)
    # pygame.quit()

from baselines.common.atari_wrappers import FrameStack

# Standard stack when env used by learner. DEPRECATED
def make_env(level=0, frame_size=96):
    """
    Create an environment with some standard wrappers.
    """
    raise DeprecationWarning("Oldie")
    env = GDoomEnv(level=level)
    env = GRewardScaler(env, scale=1)
    env = GPreprocessFrame(env,size=frame_size)
    env = FrameStack(env, 4)
    return env

# see __init__ for how this is registered in the gym
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
    # from gym.utils.play import play
    import numpy as np
    import train

    # Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
    genv = WGDoomEnv(level=2, frame_size=89)
    genv.reset() ##It is already done in the init, i do not understant the need here
    a, _, _, _ = genv.step(0)
    print( np.asarray(a).shape )

    # Also make a GPU environment, but using openai:
    env_cpu = gym.make("doom_scenario2_96-v0")
    frame = env_cpu.reset()
    print("Frame size for cpu player: ", np.asarray(frame).shape)

    env_human = gym.make("doom_scenario2_human-v0")
    frame = env_human.reset()
    print("Frame size for homan player: ", np.asarray(frame).shape)

    # env2 = SetPlayingMode(target_mode=HUMAN)(env2)
    isPlay = True
    if isPlay:
        play(env_cpu, fps=32)
    else:
        train.train(scenario = "defend_the_center", memory_size = 1000, stack_size = 4, batch_size = 64, resize = (120, 160), possible_action = getPossibleAction("defend_the_center"), game = genv.game)