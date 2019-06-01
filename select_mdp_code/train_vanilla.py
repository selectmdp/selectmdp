import tensorflow as tf
import numpy as np
import random
import config
import time

FLAGS = config.flags.FLAGS

from evaluation import Evaluator

import os
import copy as cp

# tensor option
gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3

def train():
    """main function for training
    """
    print("N_EQUIVARIANT is {}".format(FLAGS.N_EQUIVARIANT))
    print("N_INVARIANT is {}".format(FLAGS.N_INVARIANT))
    print("N_CHOOSE is {}".format(FLAGS.N_CHOOSE))
    print('Number of training episodes is', FLAGS.TRAIN_EPISODE)
    print('Number of training stexfdsfe per each episode is', FLAGS.TRAIN_STEP)

    save_name, short_name = fname(save_time=True)

    # Create directory to save results
    train_result_pth = './{}'.format(FLAGS.ENVIRONMENT)
    if not os.path.exists(train_result_pth):
        os.makedirs(train_result_pth)
    train_result_pth = os.path.join(train_result_pth, 'train-result')
    if not os.path.exists(train_result_pth):
        os.makedirs(train_result_pth)
    train_result_pth = os.path.join(train_result_pth, save_name)
    if not os.path.exists(train_result_pth):
        os.makedirs(train_result_pth)

    now = time.localtime()
    s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    train_result_pth += '/seed_' + str(FLAGS.SEED)
    print('Train result path is {}'.format(train_result_pth))

    # save params
    params_path = './{}/trained-params/'.format(FLAGS.ENVIRONMENT)
    params_path += save_name + '/seed_' + str(FLAGS.SEED) +'/'
    
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    print('Trained parameters are saved in {}'.format(params_path))

    # Global TF session
    sess = tf.Session(config=gpu_config)

    # Creating workers and corresponding evaluators
    env = Environment(FLAGS.N_EQUIVARIANT,FLAGS.N_INVARIANT, train_mode=True)
    print("create env")
    agent = Agent('global', sess)
    print("create agent")
    evaluator = Evaluator(sess, 1, train_result_pth, short_name)  # 1 means evaluator for training
    print("create evaluator")

    # generate saver for only main
    if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_main'):
        saver = tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_main'))

    sess.run(tf.global_variables_initializer())

    # copy the main params to targets
    agent.copy_parameters()

    start_time = time.time()
    
    # Learning
    for episode in range(FLAGS.RELOAD_EP*FLAGS.TRAIN_EPISODE,
        (FLAGS.RELOAD_EP+1)*FLAGS.TRAIN_EPISODE):
    
        # Reset environment
        env.reset()

        print("\n-------------- EPISODE {} --------------\n".format(episode))

        # Disable test mode before training
        agent.disable_test_mode()

        # train mode
        for step in range(FLAGS.TRAIN_STEP):
            # Normal Process
            state = cp.deepcopy(env.get_state())
            action = cp.deepcopy(agent.act(state))
            reward = cp.deepcopy(env.step(action))
            next_state = cp.deepcopy(env.get_state())

            if FLAGS.SORTED==1:
                env._sort_state()
            if FLAGS.SORTED==2:
                env._shuffle_state()

            # Agent gets reward and next state
            agent.receive_reward(reward)

            # get the loss, q-values of the current agents
            losses = cp.deepcopy(agent.get_loss())
            q_values = cp.deepcopy(agent.get_q_value())

            evaluator.save_temp_list(reward,losses,q_values)

            # with some SAVE_PERIOD, evaluator update the long term logs and preserve the consecutive transitions with SAVE_REPEAT
            if (FLAGS.TRAIN_STEP * episode + step+1) % FLAGS.SAVE_PERIOD == 0:
                evaluator.average_status()
                evaluator.save_avg_to_tensorboard(episode,step)
        
        if episode % max(int(FLAGS.TRAIN_EPISODE * FLAGS.TOTAL_RELOAD/100.0), 1)  ==0: # test 100 times
            # Enable test mode
            agent.enable_test_mode()

            reward_test = 0
            losses_test = 0
            q_values_test = np.zeros(FLAGS.N_CHOOSE)
            check_test_start = time.time()
            repeat_test = min(int(10000/FLAGS.TRAIN_STEP), 20)

            for _ in range(repeat_test):
              env.reset()
              for step in range(FLAGS.TRAIN_STEP): #test mode
                  # Normal Process
                  state = cp.deepcopy(env.get_state())
                  action = cp.deepcopy(agent.act(state))
                  reward= cp.deepcopy(env.step(action))
                  next_state = cp.deepcopy(env.get_state())
  
                  # get the loss, q-values of the current agents
                  reward_test += reward
                  losses_test += cp.deepcopy(agent.get_loss())
                  q_values_test += cp.deepcopy(agent.get_q_value())
                      
              # save test result in tb
              avg_reward_test = reward_test/(float(FLAGS.TRAIN_STEP)*repeat_test)
              avg_losses_test = losses_test/(float(FLAGS.TRAIN_STEP)*repeat_test)
              avg_q_values_test = q_values_test/(float(FLAGS.TRAIN_STEP)*repeat_test)
              check_test_end = time.time()
              spend_time_test = check_test_end-check_test_start
            
            print('time_test', spend_time_test/repeat_test)
            evaluator.save_test_tb(avg_reward_test,avg_losses_test, avg_q_values_test, spend_time_test, episode)
            evaluator._reset() # just for clean training time

    if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_main'):
        saver.save(sess, params_path, global_step=(FLAGS.RELOAD_EP+1) * FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE)

    end_time = time.time()
    print('Time taken for training: {} seconds'.format(end_time - start_time))
    # TODO: save the time to csv

    time.sleep(5)

def set_seed():
    # Set random seed
    if FLAGS.SET_SEED:
        Seed = FLAGS.SEED + FLAGS.RELOAD_EP * 100 # for different seed
        print('Setting seed values to', Seed)
        np.random.seed(Seed)
        tf.set_random_seed(Seed)
        random.seed(Seed)

def fname(save_time=False):
    """Create file name used for saving
    """
    name = FLAGS.AGENT
    name += '-VANILLA'

    if FLAGS.SORTED==1:
        assert (FLAGS.AGENT == "dqn") or (FLAGS.AGENT == "dummy")
        print("[WARNING] Using sorted environment")
        name += '-sort'

    if FLAGS.SORTED==2:
        assert (FLAGS.AGENT == "dqn") or (FLAGS.AGENT == "dummy")
        print("[WARNING] Using sorted environment")
        name += '-shuffle'

    name += '_Eq-{}'.format(FLAGS.N_EQUIVARIANT)
    name += '_In-{}'.format(FLAGS.N_INVARIANT)
    name += '_K-{}'.format(FLAGS.N_CHOOSE)
    name += '_Ep-{}'.format(FLAGS.TRAIN_EPISODE * FLAGS.TOTAL_RELOAD)
    name += '_Step-{}'.format(FLAGS.TRAIN_STEP)

    short_name = cp.deepcopy(name)

    # if not FLAGS.NETWORK == 'NONE':
    name += '_Lr-%s' % (FLAGS.LEARNING_RATE)
    name += '_Buf-%s' % (FLAGS.BUFFER_SIZE)
    name += '_Bat-%s' % (FLAGS.BATCH_SIZE)
    name += '_NwrkEx-{}'.format(FLAGS.NWRK_EXPAND)
    name += '_Layers-{}'.format(FLAGS.LAYERS)
    name += '_Xav-{}'.format(FLAGS.XAVIER)

    return name, short_name

if __name__ == '__main__':

    # Set random seed
    set_seed()

    # Import environment
    if FLAGS.ENVIRONMENT == 'circle_env':
        assert FLAGS.N_SUBACT == 1
        assert FLAGS.N_FEATURES == 3
        from environments.circle_env import Environment
        print('circle_env is used')
    else:
        raise Exception("Undefined environment: {}".format(FLAGS.ENVIRONMENT))
    
    # Choose an agent
    if FLAGS.AGENT == 'dqn':
        from agents.dqn_vanilla import Agent
        print("dqn_vanilla agent is loaded")
    else:
        raise Exception("Undefined agent: {}".format(FLAGS.AGENT))

    # Start training
    train()
