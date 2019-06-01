import tensorflow as tf
import numpy as np
import random
import config
import time

FLAGS = config.flags.FLAGS

from evaluation import Evaluator

import os
import copy as cp

# # tensor option
gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.GPU_OPTION

N_CHOOSE = FLAGS.N_CHOOSE
def train():
    """main function for training
    """
    print("N_EQUIVARIANT is {}".format(FLAGS.N_EQUIVARIANT))
    print("N_INVARIANT is {}".format(FLAGS.N_INVARIANT))
    print("N_CHOOSE is {}".format(FLAGS.N_CHOOSE))
    print('Number of training episodes is', FLAGS.TRAIN_EPISODE)
    print('Number of training stexfdsfe per each episode is', FLAGS.TRAIN_STEP)

    save_name, short_name = fname(save_time=True)
    # print('save_name', save_name)
    # print('short_name', short_name)

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
    # print('Train result path is {}'.format(train_result_pth))

    # save params
    params_path = './{}/trained-params/'.format(FLAGS.ENVIRONMENT)
    params_path += save_name + '/seed_' + str(FLAGS.SEED) +'/'
    
    if not os.path.exists(params_path):
        os.makedirs(params_path)

    print('Trained parameters are saved in {}'.format(params_path))

    # Global TF session
    sess = tf.Session(config= gpu_config)
    # sess = tf.Session()


    # generate Features
    FEATURE_SIZEs = []
    FEATURE_SIZEs.append(np.array([FLAGS.N_FEATURES,
        FLAGS.N_FEATURES+FLAGS.N_SUBACT, FLAGS.N_FEATURES], np.int32))
    N_LAYERs = FLAGS.LAYERS
    for _ in range(1, N_LAYERs-1):
        FEATURE_SIZEs.append(FEATURE_SIZEs[0] * FLAGS.NWRK_EXPAND)
    FEATURE_SIZEs.append(np.array([0,0,FLAGS.N_SUBACT], np.int32))

    ELEMENT_SIZEs = [FLAGS.N_INVARIANT, FLAGS.N_CHOOSE, FLAGS.N_EQUIVARIANT]

    # Create main and target network    
    main_params = []
    target_params = []

    # separate params case
    if FLAGS.NETWORK == "DIFFEINET":
        print("DIFFEINET")
        from agents.q_network.params_einet import Params
        Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)
        for i in range(FLAGS.N_CHOOSE):
            main_params.append(Param_generator.generate_layers(i, True))
            target_params.append(Param_generator.generate_layers(i, False))

    elif FLAGS.NETWORK == "PROGRESSIVE":
        print("PROGRESSIVE")
        param_progr_list =[0,N_CHOOSE]
        param_progr_set = set(param_progr_list)
        
        LOG_N_CHOOSE = int(np.ceil(np.log2(N_CHOOSE)))
        list_num = min([int(np.floor(LOG_N_CHOOSE * FLAGS.RELOAD_EP/(FLAGS.RATIO_PROGRESSIVE * FLAGS.TOTAL_RELOAD))), LOG_N_CHOOSE])
        # print(list_num)
        from agents.q_network.params_einet import Params
        Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)

        
        main_params_cands = []
        target_params_cands = []
        for i in range(N_CHOOSE):
            main_params_cands.append(Param_generator.generate_layers(i, True))
            target_params_cands.append(Param_generator.generate_layers(i, False))

        
        for i in range(list_num):
            # param_progr_set = set(list(param_progr_list))
            for j in range(len(param_progr_list)-1):
                # param_progr_set.add(param_progr_list[j])
                param_progr_set.add(int(np.floor((param_progr_list[j]+param_progr_list[j+1]+1)/2)))
                print(param_progr_set)
            param_progr_list= list(param_progr_set)
            print(param_progr_list)
            
        param_progr_list = np.array(param_progr_list)
        param_progr_list.sort()


        for i in range(len(param_progr_list)-1):
            main_params_cand = main_params_cands[param_progr_list[i]]
            target_params_cand = target_params_cands[param_progr_list[i]]
            for j in range(param_progr_list[i], param_progr_list[i+1]):
                main_params.append(main_params_cand)
                target_params.append(target_params_cand)

    elif FLAGS.NETWORK == "PROGRESSIVE_1_K":
        print("PROGRESSIVE_1_K")
        param_progr_list =[0,N_CHOOSE]
        param_progr_set = set(param_progr_list)
        
        LOG_N_CHOOSE = int(np.ceil(np.log2(N_CHOOSE)))

        if FLAGS.RELOAD_EP >= 0.5*FLAGS.RATIO_PROGRESSIVE * FLAGS.TOTAL_RELOAD:
            param_progr_list = list(np.arange(N_CHOOSE+1))

        
        # print(list_num)
        from agents.q_network.params_einet import Params
        Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)

        
        main_params_cands = []
        target_params_cands = []
        for i in range(N_CHOOSE):
            main_params_cands.append(Param_generator.generate_layers(i, True))
            target_params_cands.append(Param_generator.generate_layers(i, False))

        param_progr_list = np.array(param_progr_list)
        print(param_progr_list)
        for i in range(len(param_progr_list)-1):
            main_params_cand = main_params_cands[param_progr_list[i]]
            target_params_cand = target_params_cands[param_progr_list[i]]
            for j in range(param_progr_list[i], param_progr_list[i+1]):
                main_params.append(main_params_cand)
                target_params.append(target_params_cand)
        # print(main_params)
        # print(target_params)

    elif FLAGS.NETWORK == "PROGRESSIVE_ROOT":
        print("PROGRESSIVE_ROOT")
        param_progr_list =[0,N_CHOOSE]
        

        if FLAGS.RELOAD_EP >= 0.5*FLAGS.RATIO_PROGRESSIVE * FLAGS.TOTAL_RELOAD:
            param_progr_list = [0, int(np.sqrt(N_CHOOSE)), N_CHOOSE]

        if FLAGS.RELOAD_EP >= 1*FLAGS.RATIO_PROGRESSIVE * FLAGS.TOTAL_RELOAD:
            param_progr_list = list(np.arange(N_CHOOSE+1))
        
        # print(list_num)
        from agents.q_network.params_einet import Params
        Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)
        
        main_params_cands = []
        target_params_cands = []
        for i in range(N_CHOOSE):
            main_params_cands.append(Param_generator.generate_layers(i, True))
            target_params_cands.append(Param_generator.generate_layers(i, False))

        param_progr_list = np.array(param_progr_list)
        print(param_progr_list)
        for i in range(len(param_progr_list)-1):
            main_params_cand = main_params_cands[param_progr_list[i]]
            target_params_cand = target_params_cands[param_progr_list[i]]
            for j in range(param_progr_list[i], param_progr_list[i+1]):
                main_params.append(main_params_cand)
                target_params.append(target_params_cand)

    # for sharingSHAREEINET
    elif FLAGS.NETWORK == "SHAREEINET":
        print("SHAREEINET")
        from agents.q_network.params_einet import Params
        Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)
        main_param = Param_generator.generate_layers('share', True)
        target_param = Param_generator.generate_layers('share', False)
        for _ in range(FLAGS.N_CHOOSE):
            main_params.append(main_param)
            target_params.append(target_param)

    # for vanilla
    elif FLAGS.NETWORK == "VANILLA":
        print("VANILLA")
        for i in range(FLAGS.N_CHOOSE):
            from agents.q_network.params_vanilla import Params
            Param_generator = Params('global', FEATURE_SIZEs, ELEMENT_SIZEs)
            main_params.append(Param_generator.generate_layers(i, True))
            target_params.append(Param_generator.generate_layers(i, False))

    elif FLAGS.NETWORK == "NONE":
        print("Not using neural network")

    else:
        raise Exception("Undefined FLAGS.NETWORK: {}".format(FLAGS.NETWORK))

    print('length main_params', len(main_params))

    # generate saver for only main  
    if tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_main'):
        saver = tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_main'))

    # Creating workers and corresponding evaluators
    # print('init_einet make', main_params[-1][0][0][-1][-1], sess.run(main_params[-1][0][0][-1][-1]))
    # print("create env")
    env = Environment(FLAGS.N_EQUIVARIANT,FLAGS.N_INVARIANT, train_mode=True)
    # print("create agent")

    agent = Agent(FLAGS.AGENT, sess, FEATURE_SIZEs, main_params, target_params)
    # print("create evaluator")
    evaluator = Evaluator(sess, 1, train_result_pth, short_name)  # 1 means evaluator for training

    sess.run(tf.global_variables_initializer())

    # print('before load main_param', main_param[0][0][-1][-1], sess.run(main_param[0][0][-1][-1]))
    # print('befoexre load agent.q_network.main_param', agent.q_network.main_params[0][0][0][-1][-1], sess.run(agent.q_network.main_params[0][0][0][-1][-1]))

    # reload the params if start ep is not 0
    current_step=(FLAGS.RELOAD_EP)*FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE
    if not (FLAGS.RELOAD_EP-FLAGS.MAX_TO_KEEP) % 50 == 0 :
        past_step = (FLAGS.RELOAD_EP-FLAGS.MAX_TO_KEEP)*FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE
        params_name = params_path + '-' + str(past_step)
        params_name_list = ['.data-00000-of-00001', '.index', '.meta']
        for params_element in params_name_list:
            if(os.path.isfile(params_name+params_element)):
                os.remove(params_name+params_element)
            # print("File Exists!!")

    if not FLAGS.RELOAD_EP == 0:
        saver.restore(sess, params_path + '-' + str(current_step))
        # print('after load main_param', main_param[0][0][-1][-1], sess.run(main_param[0][0][-1][-1]))
        # print('after load agent.q_network.main_param', agent.q_network.main_params[0][0][0][-1][-1], sess.run(agent.q_network.main_params[0][0][0][-1][-1]))

        if not FLAGS.NETWORK == "NONE":
            agent.q_network.replay_buffer.load_buffer(params_path,current_step)
            # print('buffer', agent.q_network.replay_buffer.total_bufs[2])


    # copy the main params to targets
    if not FLAGS.NETWORK == "NONE":
        agent.q_network.copy_target()


    # for i in range(N_CHOOSE):
    #     # print('after start repeat main_param'+str(i), main_params_cands[i][0][0][-1][-1], sess.run(main_params_cands[i][0][0][-1][-1]))

    #     print('real updated main networks'+str(i), sess.run(main_params[i][0][0][-1][-1]))
    #     print('real updated target networks'+str(i), sess.run(target_params[i][0][0][-1][-1]))

    # print('after laod, agent.q_network',agent.q_network.main_params[-1][0][0][-1][-1],  sess.run(agent.q_network.main_params[-1][0][0][-1][-1]))
    # print('after laod, main_params',main_params[-1][0][0][-1][-1],  sess.run(main_params[-1][0][0][-1][-1]))

    # print('graphkey', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # print(agent.q_network.replay_buffer.total_bufs[0][0])
    # Initialize variables
    # sess.run(tf.global_variables_initializer())

    # Copy weight values from global network to local workers
    # agent.copy_parameters() #TODO: without sync update its empty

    # print('after copy, q_network',agent.q_network.main_params[-1][0][0][-1][-1],  sess.run(agent.q_network.main_params[-1][0][0][-1][-1]))
    # print('after copy, main_params',main_params[-1][0][0][-1][-1],  sess.run(main_params[-1][0][0][-1][-1]))

    # exit(0)

    start_time = time.time()
    
    # evaluator.open_csvs()
    # Learning
    for episode in range(FLAGS.RELOAD_EP*FLAGS.TRAIN_EPISODE,
     (FLAGS.RELOAD_EP+1)*FLAGS.TRAIN_EPISODE):
    
        # print('after start repeat main_param',  main_param[0][0][-1][-1], sess.run(main_param[0][0][-1][-1]))
        # print('after start repeat agent.q_network.main_param',  agent.q_network.main_params[0][0][0][-1][-1], sess.run(agent.q_network.main_params[0][0][0][-1][-1]))

        # Reset environment
        env.reset()

        # print("\n-------------- EPISODE {} --------------\n".format(episode))

        # train mode
        for step in range(FLAGS.TRAIN_STEP):
            # Normal Process

            if FLAGS.SORTED==1:
                # state = cp.deepcopy(env.get_state())
                # print('before sort state', state)
                env._sort_state()
            if FLAGS.SORTED==2:
                env._shuffle_state()

            state = cp.deepcopy(env.get_state())
            # print('state-1', state)
            # print('after sort state', state)
            # if step == 2:
            #     exit(0)
            action = cp.deepcopy(agent.act(state))
            reward = cp.deepcopy(env.step(action))
            
            
            if FLAGS.SORTED==1:
                # state = cp.deepcopy(env.get_state())
                # print('before sort state', state)
                env._sort_state()
            if FLAGS.SORTED==2:
                env._shuffle_state()
            next_state = cp.deepcopy(env.get_state())

            # Agent gets reward and next state
            agent.receive_reward(reward)
            agent.receive_next_state(next_state)
            # print('experience', state, action, reward, next_state)
            # print('state', state)
            # print('action', action)
            # print('next_state', next_state)

            # get the loss, q-values of the current agents
                
            losses = cp.deepcopy(agent.get_loss())
            q_values = cp.deepcopy(agent.get_q_value())

            evaluator.save_temp_list(reward,losses,q_values)

            # with some SAVE_PERIOD, evaluator update the long term logs and preserve the consecutive transitions with SAVE_REPEAT
            # if step % FLAGS.SAVE_PERIOD < FLAGS.SAVE_REPEAT:
            if (FLAGS.TRAIN_STEP * episode + step+1) % FLAGS.SAVE_PERIOD == 0:

                # pointer = step % FLAGS.SAVE_PERIOD
                # if pointer == 0: # average the status
                evaluator.average_status()
                evaluator.save_avg_to_tensorboard(episode,step)

                # # save some results
                # if FLAGS.N_INVARIANT: # Inv feature exist
                #     trans_list = [episode, step, np.round(state['equivariant_array'],2),np.round(state['invariant_array'],2),action,
                #         np.round(reward,2),np.round(next_state['equivariant_array'],2),np.round(next_state['invariant_array'],2),
                #         np.round(losses,2),np.round(q_values,2)]
                #     evaluator.get_transition(trans_list, pointer)
                # else: # otherwise
                #     trans_list = [episode, step, np.round(state['equivariant_array'],2), 0.000 ,action,
                #         np.round(reward,3),np.round(next_state['equivariant_array'],2), 0.000,
                #         np.round(losses,2),np.round(q_values,2)]
                #     evaluator.get_transition(trans_list, pointer)

                # write the csv file both averaged status, and REPEAT_SAVE Consecutive transitions after it log whole csvs
                # if (pointer +1) == FLAGS.SAVE_REPEAT:
                    # TODO: make the test mode
                    # evaluator.save_avg_to_csv()
        
            if  (FLAGS.TRAIN_STEP * episode + step+1) % (int(FLAGS.TOTAL_RELOAD*FLAGS.TRAIN_STEP*FLAGS.TRAIN_EPISODE)/100) == 0: # test 100 times
                reward_test = 0
                losses_test = 0
                q_values_test = np.zeros(FLAGS.N_CHOOSE)
                check_test_start = time.time()
                repeat_test = min(int(25000/FLAGS.TRAIN_STEP), 20)
                for _ in range(repeat_test):
                    env.reset()
                    if FLAGS.ENVIRONMENT == "predator_prey_discrete":
                        test_check=True
                        env.reset(test_check)
                    for step in range(FLAGS.TRAIN_STEP): #test mode
                        # Normal Process
                        # print('step', step)


                        if FLAGS.SORTED==1:
                            env._sort_state()
                        if FLAGS.SORTED==2:
                            env._shuffle_state()


                        state = cp.deepcopy(env.get_state())

                        action = cp.deepcopy(agent.act(state, train=False))
                        reward= cp.deepcopy(env.step(action))

                        if FLAGS.SORTED==1:
                            env._sort_state()
                        if FLAGS.SORTED==2:
                            env._shuffle_state()
                            
                        next_state = cp.deepcopy(env.get_state())
        
                        # Agent gets reward and next state
                        agent.receive_reward(reward)
                        agent.receive_next_state(next_state, train=False)
        
                        # get the loss, q-values of the current agents
                            
                        reward_test += reward
                        losses_test += agent.get_loss()
                        q_values_test += agent.get_q_value()
        
                        # # evaluator gets the status to be averaged 
                        # evaluator.save_test_list(reward,losses_test,q_values_test)
                        
                # save test result in tb
                    print('reward', reward_test)

                    avg_reward_test = reward_test/float(float(FLAGS.TRAIN_STEP)*repeat_test)
                    if FLAGS.ENVIRONMENT =="predator_prey_discrete":
                        print('avg_reward', avg_reward_test)
                        lower_reward = max(0.001, np.float(avg_reward_test *FLAGS.N_CHOOSE))
                        avg_reward_test = FLAGS.N_INVARIANT/float(lower_reward)
                    avg_losses_test = losses_test/(float(FLAGS.TRAIN_STEP)*repeat_test)
                    avg_q_values_test = q_values_test/(float(FLAGS.TRAIN_STEP)*repeat_test)
                    check_test_end = time.time()
                    spend_time_test = check_test_end-check_test_start
                
                print('time_test', spend_time_test/repeat_test)
                evaluator.save_test_tb(avg_reward_test,avg_losses_test, avg_q_values_test, spend_time_test, episode)
                
                evaluator._reset() # just for clean training time

    # save parameters
    # if episode % max(int((FLAGS.TRAIN_EPISODE-FLAGS.BUFFER_SIZE)/10),1)==0:

    # print('before save main_param', main_param[0][0][-1][-1], sess.run(main_param[0][0][-1][-1]))
    # print('before save agent.q_network.main_param', agent.q_network.main_params[0][0][0][-1][-1], sess.run(agent.q_network.main_params[0][0][0][-1][-1]))

    #TODO: copy the parameters with the consideration of the lists
    if FLAGS.NETWORK == "PROGRESSIVE" or FLAGS.NETWORK == "PROGRESSIVE_1_K" or FLAGS.NETWORK == "PROGRESSIVE_ROOT":
        for i in range(len(param_progr_list)-1):
            for j in range(param_progr_list[i], param_progr_list[i+1]):
                # try:
                sess.run([_get_copy_ops('global_main_tran_'+str(param_progr_list[i])+'_', 'global_main_tran_'+str(j)+'_')])
                # except:
                #     print('i,j',i,j)
                #     print('param_progr_list[i]',param_progr_list[i])


    # for i in range(N_CHOOSE):
    #     print('after start repeat main_param'+str(i), main_params_cands[i][0][0][-1][-1], sess.run(main_params_cands[i][0][0][-1][-1]))
    # print('after start repeat agent.q_network.main_param',  agent.q_network.main_params[0][0][0][-1][-1], sess.run(agent.q_network.main_params[0][0][0][-1][-1]))

    # for i in range(N_CHOOSE):
        
    #     from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_main_tran_'+str(i))
    #     print(len(from_vars))
    #     print(from_vars)



    if tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_main'):
        saver.save(sess, params_path, global_step=(FLAGS.RELOAD_EP+1) * FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE)
        

    if not FLAGS.NETWORK == 'NONE':
        agent.q_network.replay_buffer.save_buffer(params_path)
    # finish csv file and draw the figure  
    # evaluator.close_csvs()      
    # evaluator.end_save_result()

    end_time = time.time()
    print('Time taken for training: {} seconds'.format(end_time - start_time))
    # TODO: save the time to csv

    time.sleep(5)

def set_seed():
    # Set random seed
    if FLAGS.SET_SEED:
        Seed = FLAGS.SEED + FLAGS.RELOAD_EP * 100 # for different seed
        # Seed = FLAGS.SEED + FLAGS.RELOAD_EP * 100
        print('Setting seed values to', Seed)
        np.random.seed(Seed)
        tf.set_random_seed(Seed)
        random.seed(Seed)

def fname(save_time=False):
    """Create file name used for saving
    """
    name = FLAGS.AGENT


    name += '-{}'.format(FLAGS.NETWORK)

    if FLAGS.SORTED==1:
        assert (FLAGS.AGENT == "dqn") or (FLAGS.AGENT == "dummy") or (FLAGS.AGENT == "greedy_good") or (FLAGS.AGENT=="CENTRAL_GREEDY") or (FLAGS.AGENT=="dqn_ind") or (FLAGS.AGENT=="dqn_individual")
        print("[WARNING] Using sorted environment")
        name += '-sort'

    if FLAGS.SORTED==2:
        assert (FLAGS.AGENT == "dqn") or (FLAGS.AGENT == "dummy") or (FLAGS.AGENT=="CENTRAL_GREEDY")
        print("[WARNING] Using sorted environment")
        name += '-shuffle'

    name += '_Eq-{}'.format(FLAGS.N_EQUIVARIANT)
    name += '_In-{}'.format(FLAGS.N_INVARIANT)
    name += '_K-{}'.format(FLAGS.N_CHOOSE)
    name += '_Ep-{}'.format(FLAGS.TRAIN_EPISODE * FLAGS.TOTAL_RELOAD)
    name += '_Step-{}'.format(FLAGS.TRAIN_STEP)
    name += '_Command-{}'.format(FLAGS.N_SUBACT)
    # name += '_save-{}'.format(FLAGS.SAVE_PERIOD)
    # name += '_sorted-{}'.format(FLAGS.SORTED)

    short_name = cp.deepcopy(name)

    # if not FLAGS.NETWORK == 'NONE':
    name += '_Lr-%s' % (FLAGS.LEARNING_RATE)
    name += '_Buf-%s' % (FLAGS.BUFFER_SIZE)
    name += '_Bat-%s' % (FLAGS.BATCH_SIZE)
    # name += '_TargetFr-%s' % (FLAGS.UPDATE_TARGET)
    name += '_NwrkEx-{}'.format(FLAGS.NWRK_EXPAND)
    name += '_Layers-{}'.format(FLAGS.LAYERS)
    name += '_Xav-{}'.format(FLAGS.XAVIER)

    if FLAGS.ENVIRONMENT == "predator_prey_discrete":
        name += '_Grid-{}'.format(FLAGS.GRID_SIZE)

    # if save_time == True:
    #     # save time
    #     now = time.localtime()
    #     s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    #     name += '_{}'.format(s_time)

    return name, short_name

def _get_copy_ops(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    # print('length',len(from_vars), len(to_vars))
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

if __name__ == '__main__':

    # Set random seed
    set_seed()

    # Import environment
    if FLAGS.ENVIRONMENT == 'circle_env':
        assert FLAGS.N_SUBACT == 1
        assert FLAGS.N_FEATURES == 3
        from environments.circle_env import Environment
        print('circle_env is used')
    elif FLAGS.ENVIRONMENT == 'circle_env_good':
        assert FLAGS.N_SUBACT == 1 or 5
        assert FLAGS.N_FEATURES == 3
        from environments.circle_env_good import Environment
        print('circle_env_good is used')
    elif FLAGS.ENVIRONMENT == 'predator_prey':
        assert FLAGS.N_SUBACT == 4
        assert FLAGS.N_FEATURES == 2
        from environments.predator_prey import Environment
        print('predator_prey is used')

    elif FLAGS.ENVIRONMENT == 'predator_prey_discrete':
        assert FLAGS.N_SUBACT == 5
        assert FLAGS.N_FEATURES == 2
        from environments.predator_prey_discrete import Environment
        print('predator_prey_discrete is used')
    else:
        raise Exception("Undefined environment: {}".format(FLAGS.ENVIRONMENT))
    
    # Choose an agent
    if FLAGS.AGENT == 'dummy':
        from agents.dummy import Agent
        print("dummy agent is loaded")
    elif FLAGS.AGENT == 'greedy':
        if FLAGS.ENVIRONMENT == 'circle_env':
            from agents.greedy import AgentCircle as Agent
        elif FLAGS.ENVIRONMENT == 'predator_prey':
            from agents.greedy import AgentPredPrey as Agent
        else:
            raise Exception("Undefined environment: {}".format(FLAGS.ENVIRONMENT))
        print("greedy agent is loaded")
        
    elif FLAGS.AGENT == 'greedy_good':
        if FLAGS.ENVIRONMENT == 'circle_env':
            from agents.greedy_good import AgentCircle as Agent
        elif FLAGS.ENVIRONMENT == 'circle_env_good':
            from agents.greedy_good import AgentCircle as Agent

        else:
            raise Exception("Undefined environment: {}".format(FLAGS.ENVIRONMENT))
        print("greedy agent is loaded")

    elif FLAGS.AGENT == 'dqn':
        from agents.dqn import Agent
        print("dqn agent is loaded")

    elif FLAGS.AGENT == 'dqn_myopic':
        from agents.dqn import Agent
        print("dqn_myopic agent is loaded")
        
    elif FLAGS.AGENT == 'dqn_ind':
        from agents.dqn import Agent
        print("dqn_ind agent is loaded")

    elif FLAGS.AGENT== "CENTRAL_GREEDY":
        from agents.dqn import Agent
        print("central greedy agent is loaded")

    elif FLAGS.AGENT=="dqn_individual":
        from agents.dqn import Agent
        print("dqn_individual agent is loaded")
    else:
        raise Exception("Undefined agent: {}".format(FLAGS.AGENT))

    # Start training
    train()
    # print(
    exit(0)
