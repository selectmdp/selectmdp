#!/usr/bin/env python
# coding=utf8

# import matplotlib
# matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import time
import csv
import os
import copy as cp
import sys

FLAGS = tf.flags.FLAGS
np.set_printoptions(threshold=5000)

class Evaluator:
    def __init__(self, sess, train_mode, save_path, save_name):
        """Initialize the basic values when evaluator is created
        """
        self.sess = sess
        self._init_values(train_mode,save_path, save_name)
        self._init_tbsaver()
        self._reset()

        return None

    def _init_tbsaver(self):
        """Generate the savers -tensor
        """
        # test mode saver
        self.reward_test_tb_place = tf.placeholder(tf.float32)
        self.loss_test_tb_place = tf.placeholder(tf.float32)
        self.qvalue_test_tb_place = tf.placeholder(tf.float32)
        self.time_test_tb_place = tf.placeholder(tf.float32)

        self.avg_reward_test_hist = tf.summary.scalar('reward', self.reward_test_tb_place)
        self.avg_loss_test_hist = tf.summary.scalar('loss', self.loss_test_tb_place)
        self.avg_qval_test_hist = tf.summary.scalar('q-value', self.qvalue_test_tb_place)
        self.avg_time_test_hist = tf.summary.scalar('time_test', self.time_test_tb_place)

        if FLAGS.REDUCED_TENSOR==1:
            self.tb_merged_test = tf.summary.merge([self.avg_reward_test_hist, self.avg_qval_test_hist])

        elif FLAGS.REDUCED_TENSOR==0:
            self.tb_merged_test = tf.summary.merge([self.avg_reward_test_hist, self.avg_loss_test_hist, self.avg_qval_test_hist, self.avg_time_test_hist])

        # training saver
        self.reward_tb_place = tf.placeholder(tf.float32)
        self.time_tb_place = tf.placeholder(tf.float32)
        self.qval_tb_place = tf.placeholder(tf.float32)

        self.avg_reward_hist = tf.summary.scalar('reward_train', self.reward_tb_place)
        self.avg_time_hist = tf.summary.scalar('time_train', self.time_tb_place)
        self.avg_qval_hist = tf.summary.scalar('q-value_train', self.qval_tb_place)

        # from 19/01/20: reduced mode
        if FLAGS.REDUCED_TENSOR == 1:
            self.tb_merged_train = tf.summary.merge([self.avg_reward_hist])
        elif FLAGS.REDUCED_TENSOR == 0: # ~19/01/20 old avg all
            self.tb_merged_train = tf.summary.merge([self.avg_reward_hist, self.avg_time_hist, self.avg_qval_hist])

        self.tbwriter = tf.summary.FileWriter('./'+self.save_path, self.sess.graph)

        return None

    def get_transition(self, trans_list, ith):
        """Get the status to be saved
        save trans_list to self.trans_list[ith] 
        """
        # print(trans_list)
        self.trans_list[ith] = cp.deepcopy(trans_list)

        return None

    def save_temp_list(self, reward, losses, q_values):
        """Report the temporal status and transitions to be averaged
        """
        if self.pointer_temp == 500:
            # print(reward, losses, q_values)
            pass
        self.reward_list_temp[self.pointer_temp] = reward
        self.loss_list_temp[self.pointer_temp] = losses
        self.q_value_list_temp[self.pointer_temp] = q_values
        self.pointer_temp += 1

        return None
    
    def average_status(self):
        """Average the status of the circles in temporal list.
        append to long term lists and print it to terminal
        np.reshape used for correct appending
        """
        # self.reward_list_long = np.append(self.reward_list_long, 
        #     np.reshape(np.round(np.average(self.reward_list_temp,0),3),[1,-1]))
        # self.loss_list_long = np.append(self.loss_list_long,
        #     np.reshape(np.round(np.average(self.loss_list_temp,0),2),[1,-1]), axis=0)
        # self.q_value_list_long = np.append(self.q_value_list_long,
        #     np.reshape(np.round(np.average(self.q_value_list_temp,0),2),[1,-1]), axis=0)

        # print the averaged value
        # self._print_averaged_status()

        return None

    def save_avg_to_tensorboard(self, episode, step):
        """Save the longterm logs to tensorboard
        """
        csv_data = [np.average(self.reward_list_temp,0),
                    np.average(self.loss_list_temp,0).tolist(),
                    np.average(self.q_value_list_temp,0).tolist()]
        avg_reward = csv_data[0]
        avg_q_value = csv_data[-1][-1]
        # print('avg_reward', avg_reward)
        # print('avg_q_value', avg_q_value)

        time_consumed= time.time() - self.check_time
        # print('time_consumed', time_consumed)
        summary = self.sess.run(self.tb_merged_train, {self.reward_tb_place: avg_reward, self.time_tb_place: time_consumed, self.qval_tb_place: avg_q_value})

        self.tbwriter.add_summary(summary, episode*FLAGS.TRAIN_STEP+step)

        self._reset()

    def save_avg_to_csv(self):
        """Save the longterm logs to the csv file
        and the current transitions
        """
        csv_data = [self.reward_list_long[-1],
                    self.loss_list_long[-1].tolist(),
                    self.q_value_list_long[-1].tolist()]

        # save longterm csv
        self._wrt_csv_row(self.wr_avg, csv_data)
       
        # save the transed transitions
        # for i in range(FLAGS.SAVE_REPEAT):
        #     self._wrt_csv_row(self.wr_trans, self.trans_list[i])
        
        self._reset()

        return None
    
    def open_csv(self, path):
        """Open the csv file, if there is no directory, create it
        Return: wr: writer, fcsv: csv file naming 
        """
        # path = self.save_path+'/'+self.f_name()+'.csv'
        if sys.version_info[0] < 3: # python 2
            if os.path.exists(path):
                fcsv = open(path , 'ab')
            else:
                fcsv = open(path, 'wb')

        else: # python 3
            if os.path.exists(path):
                fcsv = open(path , 'a', newline = '')
            else:
                fcsv = open(path, 'w', newline = '')

        wr = csv.writer(fcsv)

        return fcsv, wr

    def end_save_result(self):
        """End the save file.
        draw the figures
        """
        self.close_csvs()
        self._save_result_figure()

        return None

    def _reset(self):
        """Reset the temporal lists for each FLAGS.SAVE_PERIOD
        """
        # temporal term list to be averaged
        self.reward_list_temp = np.zeros(FLAGS.SAVE_PERIOD)
        self.loss_list_temp = np.zeros((FLAGS.SAVE_PERIOD, FLAGS.N_CHOOSE))
        self.q_value_list_temp = np.zeros((FLAGS.SAVE_PERIOD, FLAGS.N_CHOOSE))
        self.pointer_temp = 0 # pointer to be saved

        self.check_time = time.time()

        # temp term list to be written in csv
        self.trans_list = FLAGS.SAVE_REPEAT * [None]
        
        return None

    def save_test_tb(self, avg_reward_test, avg_loss_test, avg_q_value_test, time, episode):
        """Save the result for test mode
        """
        avg_reward = avg_reward_test
        avg_loss = avg_loss_test[-1]
        avg_q_value = avg_q_value_test[-1]

        summary = self.sess.run(self.tb_merged_test, {self.reward_test_tb_place: avg_reward, self.loss_test_tb_place: avg_loss, self.qvalue_test_tb_place: avg_q_value, self.time_test_tb_place: time})

        self.tbwriter.add_summary(summary, episode*FLAGS.TRAIN_STEP)


    def _init_values(self, train_mode, save_path, save_name):
        """Initialize the important arrays or values for the evaluations
        """
        # check for train_mode: do not used currently
        self.train_mode = train_mode
        
        # name to save directory
        self.save_path = save_path
        self.save_name = save_name
        self.save_path_avg_csv = self.save_path+'/'+'average.csv'
        self.save_path_trans_csv = self.save_path+'/'+'transitions.csv'

        # reset the temporal lists for reward, loss, q_values
        self.reward_list_temp = np.zeros(FLAGS.SAVE_PERIOD)
        self.loss_list_temp = np.zeros((FLAGS.SAVE_PERIOD, FLAGS.N_CHOOSE))
        self.q_value_list_temp = np.zeros((FLAGS.SAVE_PERIOD, FLAGS.N_CHOOSE))
        self.pointer_temp = 0 # pointer for temporal values

        # long averaged term list, reshapiring is 
        self.reward_list_long = np.zeros([1,1])
        self.loss_list_long = np.zeros([1,FLAGS.N_CHOOSE])
        self.q_value_list_long = np.zeros([1,FLAGS.N_CHOOSE])

        # open csv_files_to save
        # self.fcsv_avg, self.wr_avg = self.open_csv(self.save_path_avg_csv) # save averaged status 
        # self.fcsv_trans, self.wr_trans = self.open_csv(self.save_path_trans_csv) # save transitions

        return None

    def _print_averaged_status(self):
        """Print out step, state, action, reward
        """
        print("---------------------------------------")
        print("Averaged reward: {}".format(np.average(self.reward_list_temp,0)))
        print("Loss: {}".format(np.average(self.loss_list_temp,0).tolist()))
        print("Q values: {}".format(np.average(self.q_value_list_temp,0).tolist()))
        print("---------------------------------------")
        return None

    def _wrt_csv_row(self, wr, row):
        """Write each information of the row
        wr: writer object
        """
        wr.writerow(row)

        return None

    def open_csvs(self):
        """open the csv files
        """
        self.fcsv_avg, self.wr_avg = self.open_csv(self.save_path_avg_csv) # save averaged status 
        # self.fcsv_trans, self.wr_trans = self.open_csv(self.save_path_trans_csv) # save transitions

        return None

    def close_csvs(self):
        """Close the csv file
        """
        self.fcsv_avg.close()
        # self.fcsv_trans.close()

        return None

    def _save_result_figure(self, save_period = FLAGS.SAVE_PERIOD):
        """save the figures for each stats by using self.sub_figure() 
        cut the first value due to the zero_initial start.
        """
        # save average reward
        # self._save_result_subfigure(self.reward_list_long[1:], "Reward")

        # # repeatedly save losses and q_values for each sub-actions
        # for i in range(FLAGS.N_CHOOSE):
        #     self._save_result_subfigure(self.loss_list_long[1:,i], "Loss",i)
        #     self._save_result_subfigure(self.q_value_list_long[1:,i], "Q_value",i)

        # print("Successfuly save the figures.")
        
        return None

    def _save_result_subfigure(self, data, data_type, action_num = None,  save_period = FLAGS.SAVE_PERIOD):
        """Save the one type of data into one figure as 'data_type' type to 'data_type.png'
        data: each self.reward_list_long, self.loss_list_long[:,action_num], self.q_value_list_long[:,action_num]
        """
        total_plotted_numb = len(data) # total number of points
        data_name = data_type # file to save data
        if action_num:
            data_name +='_'+str(action_num) # add action_num

        plt.figure()
        plt.plot(save_period * np.arange(total_plotted_numb),  data, 'yo-', label=data_name, markersize=3)
        plt.xlabel('Step')
        plt.ylabel(data_name)
        plt.legend()
        plt.suptitle(self.save_name)

        plt.savefig(os.path.join(self.save_path, data_name +'.png'))

        return None

    # def _fname(self, save_time=False):
    #     """making the saving file name
    #     """

    #     # Agent name - used to save result
    #     agent_name = FLAGS.AGENT
    #     agent_name += '-{}'.format(FLAGS.NETWORK)

    #     if FLAGS.SORTED:
    #         assert (FLAGS.AGENT == "dqn") or (FLAGS.AGENT == "dummy")
    #         print("[WARNING] Using sorted environment")
    #         agent_name += '-sort'

    #     return str()
