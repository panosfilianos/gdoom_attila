import time
import scipy
import numpy as np

from utils.network_params import *

import tensorflow as tf

from PIL import Image


def process_frame(frame, crop, resize):
    """
    Description
    ---------------
    Crop and resize Doom screen frame.

    Parameters
    ---------------
    frame  : np.array, screen image
    crop   : tuple, top, bottom, left and right crops
    resize : tuple, new width and height

    Returns
    ---------------
    s      : np.array, screen image cropped and resized.
    """
    y2, y1, x1, x2 = crop
    s = frame[y2:y1,x1:x2]
    s = scipy.misc.imresize(s,list(resize))
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s


    # y2, y1, x1, x2 = crop
    #
    # # we are using a lazyframes FrameStack but taking frame 0 (out of 3 total frames)
    # # as seen below
    # # print(type(frame))
    # # print(frame.shape)
    # # a lot of dimensions for RGB - change
    # # s = frame[y2:y1, x1:x2, 0]
    # s = frame[y2:y1, x1:x2]
    # # this way the output is of size resize[0] * resize[1] = height * width after resizing which is
    # # the case for the NNs input also, if we want to stack frames, we might as well but we have to change
    # # the NNs input likewise
    #
    #
    # #check for efficiency
    # s = np.array(Image.fromarray(s).resize(list(resize)))
    # # s = scipy.misc.imresize(s, list(resize))
    # s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    return s

def setup_network(sess, agent):
    with sess.as_default(), sess.graph.as_default():
        # Copy the global networks weights to local network weights
        sess.run(agent.update_local_ops)
        # if params.use_curiosity: sess.run(self.update_local_ops_P)

        # Initialize buffer for training
        agent.local_AC.episode_buffer = []

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
        #we do below outside
        # agent.game.new_episode()
        agent.episode_st = time.time()
        #
        # # Initialize LSTM gates
        rnn_state = agent.local_AC.state_init
        agent.batch_rnn_state = rnn_state
        #
        # Get first state and process it
        # s = agent.get_game.get_state().screen_buffer
        # agent.episode_frames.append(s)
        # s = process_frame(s, crop, resize)
        return rnn_state


def button_combinations(scenario='basic'):
    """
    Description
    ---------------
    Returns a list of possible action for a scenario.

    Parameters
    ---------------
    scenario : String, Doom scenario to use (default='basic')

    Returns
    ---------------
    actions : list, the one-hot encoded possible actions.
    """
    actions = []
    # [move left, move right, shoot, move forward, move backwards, turn left, turn right]

    m_left_right = [[True, False], [False, True], [False, False]]  # move left and move right
    attack = [[True], [False]]
    m_forward_backward = [[True, False], [False, True], [False, False]]  # move forward and backward
    t_left_right = [[True, False], [False, True], [False, False]]  # turn left and turn right

    if scenario == 'deadly_corridor':
        actions = np.identity(7, dtype=int).tolist()
        # [move left, move right, shoot, move forward, turn left, turn right]
        actions.extend([[0, 0, 1, 0, 0, 1, 0], #shoot, turn left
                        [0, 0, 1, 0, 0, 0, 1], #shoot, turn right
                        [1, 0, 1, 0, 0, 0, 0], #move left, shoot
                        [0, 1, 1, 0, 0, 0, 0]]) #move right, shoot

    if scenario == 'basic':
        for i in m_left_right:
            for j in attack:
                actions.append(i + j)

    if scenario == 'my_way_home':
        actions = np.identity(3, dtype=int).tolist()
        actions.extend([[1, 0, 1],
                        [0, 1, 1]])

    if scenario == 'defend_the_center':
        for i in t_left_right:
            for j in attack:
                actions.append(i + j)

    if scenario == 'defend_the_line':
        for i in t_left_right:
            for j in attack:
                actions.append(i + j)

    return actions


def normalized_columns_initializer(std=1.0):
    """
    Description
    ---------------
    Tensorflow zero-mean, std weights initializer.

    Parameters
    ---------------
    std  : float, std for the normal distribution

    Returns
    ---------------
    _initializer : Tensorflow initializer
    """

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


