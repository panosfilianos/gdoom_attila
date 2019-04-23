import os
import shutil
import threading
import multiprocessing
import tensorflow as tf

from agent import *
from utils.networks import *
from gdoom_env import *

from time import sleep
from time import time

#def train_agents():
#    tf.reset_default_graph() #https://www.tensorflow.org/api_docs/python/tf/reset_default_graph
    
#    #Delete saves directory if not loading a model
#    if not params.load_model:
#        #Delete an entire directory tree
#        shutil.rmtree(params.model_path, ignore_errors=True)
#        shutil.rmtree(params.frames_path, ignore_errors=True)
#        shutil.rmtree(params.summary_path, ignore_errors=True)

#    #Create a directory to save models to
#    if not os.path.exists(params.model_path):
#        os.makedirs(params.model_path)

#    #Create a directory to save episode playback gifs to
#    if not os.path.exists(params.frames_path):
#        os.makedirs(params.frames_path)

#    with tf.device("/cpu:0"):
#        # Generate global networks : Actor-Critic and ICM
#        master_network = AC_Network(state_size, action_size, 'global') # Generate global AC network
#        if params.use_curiosity: #do not understand
#            master_network_P = StateActionPredictor(state_size, action_size, 'global_P') # Generate global AC network
        
#        # Set number of workers
#        if params.num_workers == -1:
#            num_workers = multiprocessing.cpu_count()
#        else:
#            num_workers = params.num_workers
        
#        # Create worker classes
#        workers = []
#        for i in range(num_workers):
#            #Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.ol
#            trainer = tf.train.AdamOptimizer(learning_rate=params.lr)  #paramaters like learning rate: epsilon, gamma and so on
#            workers.append(Worker(i, state_size, action_size, trainer, params.model_path))
#        saver = tf.train.Saver(max_to_keep=5)  #It saves and restores 5 checkpoints

#    with tf.Session() as sess:
#        # Loading pretrained model
#        if params.load_model == True:
#            print ('Loading Model...')
#            ckpt = tf.train.get_checkpoint_state(model_path) #Returns a CheckpointState if the state was available, None otherwise.
#            saver.restore(sess,ckpt.model_checkpoint_path) #Restore the checkopoint found before
#        else:
#            sess.run(tf.global_variables_initializer()) #Run the session initializing the default tensorflow variables

#        #Starting initialized workers, each in a separate thread.
#        coord = tf.train.Coordinator() #This class implements a simple mechanism to coordinate the termination of a set of threads.
#        worker_threads = []
#        for worker in workers:
#            worker_work = lambda: worker.work(params.max_episodes, params.gamma, sess, coord, saver)
#            t = threading.Thread(target=(worker_work))
#            t.start()
#            sleep(0.5)
#            worker_threads.append(t)
#        coord.join(worker_threads) #This call blocks until a set of threads have terminated.

def train_agents():
    tf.reset_default_graph() #https://www.tensorflow.org/api_docs/python/tf/reset_default_graph
    
    #Delete saves directory if not loading a model
    if not params.load_model:
        #Delete an entire directory tree
        shutil.rmtree(params.model_path, ignore_errors=True)
        shutil.rmtree(params.frames_path, ignore_errors=True)
        shutil.rmtree(params.summary_path, ignore_errors=True)

    #Create a directory to save models to
    if not os.path.exists(params.model_path):
        os.makedirs(params.model_path)

    #Create a directory to save episode playback gifs to
    if not os.path.exists(params.frames_path):
        os.makedirs(params.frames_path)

    with tf.device("/cpu:0"):
        # Generate global networks : Actor-Critic and ICM
        master_network = AC_Network(state_size, action_size, 'global') # Generate global AC network
        if params.use_curiosity: #do not understand
            master_network_P = StateActionPredictor(state_size, action_size, 'global_P') # Generate global AC network
        
        # Set number of workers
        if params.num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = params.num_workers
        
        # Create worker classes
        envs = []
        for i in range(num_workers):
            #Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.ol
            trainer = tf.train.AdamOptimizer(learning_rate=params.lr)  #paramaters like learning rate: epsilon, gamma and so on

            #See if it makes sense:
            envs.append(make_env(i, state_size, action_size, trainer, params.model_path))   #Look at the parameters, there is level and framesize.
        saver = tf.train.Saver(max_to_keep=5)  #It saves and restores 5 checkpoints

    with tf.Session() as sess:
        # Loading pretrained model
        if params.load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path) #Returns a CheckpointState if the state was available, None otherwise.
            saver.restore(sess, ckpt.model_checkpoint_path) #Restore the checkopoint found before
        else:
            sess.run(tf.global_variables_initializer()) #Run the session initializing the default tensorflow variables

        #Starting initialized workers, each in a separate thread.
        coord = tf.train.Coordinator() #This class implements a simple mechanism to coordinate the termination of a set of threads.
        worker_threads = []
        for env in envs:

            #Check if it respects the train function, it should be right if trais has not been changed in the while.
            worker_work = lambda: train(env, params.max_episodes, params.gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads) #This call blocks until a set of threads have terminated.