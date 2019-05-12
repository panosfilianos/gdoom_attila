import tensorflow as tf
from utils.network_params import *
from vizdoom import GameVariable
import scipy

import moviepy.editor as mpy



import time

def print_end_episode_perfs(agent):
    """
    Description
    --------------
    Print episode statistics depending on the scenario.

    """
    print(agent.v_l_array)
    try:
        avg_v_loss = sum(agent.v_l_array) / len(agent.v_l_array)
    except:
        avg_v_loss = "nan"
    try:
        avg_p_loss = sum(agent.p_l_array) / len(agent.p_l_array)
    except:
        avg_p_loss = "nan"
    try:
        avg_e_loss = sum(agent.e_l_array) / len(agent.e_l_array)
    except:
        avg_e_loss = "nan"
    print('{}|{}|{}|{}|{}|{}|{}|{}'.format(agent.episode_count, agent.last_total_kills,
                                           agent.episode_step_count, agent.episode_reward,
                                           agent.last_total_ammo2, avg_v_loss, avg_p_loss, avg_e_loss))
    return

    if params.scenario == 'basic':
        print('{}|{}|{}|{}|{}|{}|{}|{}'.format(agent.episode_count, agent.episode_kills,
                                         agent.episode_step_count, agent.episode_reward,
                                         agent.episode_ammo, avg_v_loss, avg_p_loss, avg_e_loss))
        print(
            '{}, episode #{}, ep_reward: {}, ep_curiosity: {}, av_reward:{}, av_curiosity:{}, steps:{}, time costs:{}'.format(
                agent.name, agent.episode_count, agent.episode_reward, agent.episode_curiosity,
                agent.episode_reward / agent.episode_step_count,
                agent.episode_curiosity / agent.episode_step_count, agent.episode_step_count,
                time.time() - agent.episode_st))

    if params.scenario == 'deadly_corridor':
        print(
            '{}, health: {}, kills:{}, episode #{}, ep_reward: {}, ep_curiosity: {}, av_reward:{}, av_curiosity:{}, steps:{}, time costs:{}'.format(
                agent.name, np.maximum(0, agent.last_total_health), agent.last_total_kills, agent.episode_count,
                agent.episode_reward, agent.episode_curiosity, agent.episode_reward / agent.episode_step_count,
                agent.episode_curiosity / agent.episode_step_count,
                agent.episode_step_count, time.time() - agent.episode_st))

    if params.scenario == 'defend_the_center':
        print(
            '{}, kills:{}, episode #{}, ep_reward: {}, ep_curiosity: {}, av_reward:{}, av_curiosity:{}, steps:{}, time costs:{}'.format(
                agent.name, agent.last_total_kills, agent.episode_count, agent.episode_reward, agent.episode_curiosity,
                agent.episode_reward / agent.episode_step_count, agent.episode_curiosity / agent.episode_step_count,
                agent.episode_step_count, time.time() - agent.episode_st))

    if params.scenario == 'my_way_home':
        print(
            '{}, episode #{}, ep_reward: {}, ep_curiosity: {}, av_reward:{}, av_curiosity:{}, steps:{}, time costs:{}'.format(
                agent.name, agent.episode_count, agent.episode_reward, agent.episode_curiosity,
                agent.episode_reward / agent.episode_step_count,
                agent.episode_curiosity / agent.episode_step_count, agent.episode_step_count,
                time.time() - agent.episode_st))

def get_health_reward(env):
    """
    Description
    --------------
    Health reward.

    """
    d_health = env.game.get_game_variable(GameVariable.HEALTH) - env.agent.last_total_health
    env.agent.last_total_health = env.game.get_game_variable(GameVariable.HEALTH)
    if d_health == 0:
        return 0
    elif d_health < 0:
        return -d_health
        # return -5

def get_ammo_reward(env):
    """
    Description
    --------------
    Ammo reward.

    """
    d_ammo = env.game.get_game_variable(GameVariable.AMMO2) - env.agent.last_total_ammo2
    env.agent.last_total_ammo2 = env.game.get_game_variable(GameVariable.AMMO2)
    if d_ammo == 0:
        return 0
    elif d_ammo > 0:
        return d_ammo * 0.5
    else:
        return -d_ammo * 0.5

def get_kill_reward(env):
    """
    Description
    --------------
    Kill reward.

    """
    d_kill = env.game.get_game_variable(GameVariable.KILLCOUNT) - env.agent.last_total_kills
    env.agent.last_total_kills = env.game.get_game_variable(GameVariable.KILLCOUNT)
    if d_kill > 0:
        return d_kill * 100
    return 0


def update_target_graph(from_scope, to_scope):
    """
    Description
    ---------------
    Copies set of variables from one network to the other.

    Parameters
    ---------------
    from_scope : String, scope of the origin network
    to_scope   : String, scope of the target network

    Returns
    ---------------
    op_holder  : list, variables copied.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def choose_action_index( policy, deterministic=False):
    """
    Description
    --------------
    Choose action from stochastic policy.

    Parameters
    --------------
    policy        : np.array, actions probabilities
    deterministic : boolean, whether to
    """
    if deterministic:
        return np.argmax(policy[0])
    else:
        return np.argmax(policy == np.random.choice(policy[0], p=policy[0]))


def discount(x, gamma):
    """
    Description
    ---------------
    Returns gamma-discounted cumulated values of x
    [x0 + gamma*x1 + gamma^2*x2 + ...,
     x1 + gamma*x2 + gamma^2*x3 + ...,
     x2 + gamma*x3 + gamma^2*x4 + ...,
     ...,
     xN]

    Parameters
    ---------------
    x      : list, list of values
    gamma  : float, top, bottom, left and right crops

    Returns
    ---------------
    np.array, gamma-discounted cumulated values of x
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def make_gif(images, fname, fps=24):
    """
    Description
    ---------------
    Makes gifs from list of images

    Parameters
    ---------------
    images  : list, contains all images used to creates a gif
    fname   : str, name used to save the gif


    """

    def make_frame(t):
        try:
            x = images[int(fps * t)]
        except:
            x = images[-1]

        #  WE ARE SKIPPING THE FIRST FRAME BECAUSE IT IS LAZY WITH FUCKED UP DIMENSIONS

        x = images[0]
        if (int(t == 0)):
            x = images[1]
        else:
            try:
                x = images[int(fps * t)]
            except:
                x = images[-1]
        # print(x.shape)

        # print(x)
        # print(np.array(x))
        #
        # print(len(x))
        # print(np.array(x).shape)
        # x = np.array(x)[0]
        # return_frame = x.astype(np.uint8)

        try:
            return_frame = x.astype(np.uint8)
        except:
            print("lazyframe transfer")
            #render from LazyFrame
            # x = np.array(x)[:, :, 0]
            x = x[:,:,0]
            x= np.array(x)
            return_frame = x.astype(np.uint8)
        return return_frame

    clip = mpy.VideoClip(make_frame, duration=len(images) / fps)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)

def initialize_containers(agent):
    """
    Description
    --------------
    Initialize episode containers used for tensorboard summary.

    """
    agent.episode_rewards = []
    agent.episode_curiosities = []
    agent.episode_lengths = []
    agent.episode_mean_values = []

    if params.scenario == 'deadly_corridor':
        agent.episode_kills = []
        agent.episode_health = []
        agent.episode_ammo = []

    if params.scenario == 'basic':
        agent.episode_ammo = []

    if params.scenario == 'defend_the_center':
        agent.episode_ammo = []
        agent.episode_kills = []

def update_containers(agent):
    """
    Description
    --------------
    Update episode containers used for tensorboard summary.

    """
    agent.episode_rewards.append(agent.episode_reward)
    agent.episode_lengths.append(agent.episode_step_count)
    agent.episode_mean_values.append(np.mean(agent.episode_values))

    if params.use_curiosity:
        agent.episode_curiosities.append(agent.episode_curiosity)

    if params.scenario == 'deadly_corridor':
        agent.episode_kills.append(agent.last_total_kills)
        agent.episode_health.append(np.maximum(0, agent.last_total_health))
        agent.episode_ammo.append(agent.last_total_ammo2)

    if params.scenario == 'basic':
        agent.episode_ammo.append(agent.last_total_ammo2)

    if params.scenario == 'defend_the_center':
        agent.episode_ammo.append(agent.last_total_ammo2)
        agent.episode_kills.append(agent.last_total_kills)

def update_summary(agent):
    """
    Description
    --------------
    Update tensorboard summary using episode containers.

    """
    mean_reward = np.mean(agent.episode_rewards[-params.freq_summary:])
    mean_length = np.mean(agent.episode_lengths[-params.freq_summary:])
    mean_value = np.mean(agent.episode_mean_values[-params.freq_summary:])

    if params.use_curiosity:
        mean_curiosity = np.mean(agent.episode_curiosities[-params.freq_summary:])

    summary = tf.Summary()
    summary.value.add(tag='Perf/Episode_Length', simple_value=float(mean_length))
    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
    if params.use_curiosity:
        summary.value.add(tag='Perf/Curiosity', simple_value=float(mean_curiosity))

    if params.scenario == 'deadly_corridor':
        mean_kills = np.mean(agent.episode_kills[-params.freq_summary:])
        mean_health = np.mean(agent.episode_health[-params.freq_summary:])
        mean_ammo = np.mean(agent.episode_ammo[-params.freq_summary:])
        summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))
        summary.value.add(tag='Perf/Health', simple_value=float(mean_health))
        summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))

    if params.scenario == 'basic':
        mean_ammo = np.mean(agent.episode_ammo[-params.freq_summary:])
        summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))

    if params.scenario == 'defend_the_center':
        mean_ammo = np.mean(agent.episode_ammo[-params.freq_summary:])
        mean_kills = np.mean(agent.episode_kills[-params.freq_summary:])
        summary.value.add(tag='Perf/Ammo', simple_value=float(mean_ammo))
        summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))

    summary.value.add(tag='Losses/Value Loss', simple_value=float(agent.v_l))
    summary.value.add(tag='Losses/Policy Loss', simple_value=float(agent.p_l))
    summary.value.add(tag='Losses/Entropy', simple_value=float(agent.e_l))
    summary.value.add(tag='Losses/Grad Norm', simple_value=float(agent.g_n))
    summary.value.add(tag='Losses/Var Norm', simple_value=float(agent.v_n))
    if params.use_curiosity:
        summary.value.add(tag='Losses/Inverse Loss', simple_value=float(agent.Inv_l))
        summary.value.add(tag='Losses/Forward Loss', simple_value=float(agent.Forward_l))

    agent.summary_writer.add_summary(summary, agent.episode_count)
    agent.summary_writer.flush()