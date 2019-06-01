import matplotlib
matplotlib.use('Agg') 

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import config
import time

import csv
import os
import copy as cp
import sys
import ast

FLAGS = config.flags.FLAGS

def read_csv(path):
    """Read the .csv with trans file
    -----------
    Return: wr: writer, fcsv: csv file naming 
    """        
    print('open path is ', path)
    if sys.version_info[0] < 3: # python 2
        if os.path.exists(path):
            fcsv = open(path , 'rb')
        else:
            print('there is not any file')
            exit(0)

    else: # python 3
        if os.path.exists(path):
            fcsv = open(path , 'r')
        else:
            print('there is not any file')
            exit(0)

    reader = csv.reader(fcsv)
    csv_list = list(reader) 
    
    return csv_list


def save_trans_circles(csv_list, path, fname, ep, step):
    """Save the current transitions (s,a,r,s') with the figures
    in the figure, only selected good circles, and bad circles are shown.
    --------
    Input: episode: number of episode, step: number of step
    Ouput: draw the figures
    """
    # convert the string to proper types 
    # current equivariant states 
    good_circles_ndarray = convert_str_ndarray(csv_list[2])

    # current invariant states
    bad_circles_ndarray = convert_str_ndarray(csv_list[3])

    # selected equivariant elements
    selected_good_circles = ast.literal_eval(csv_list[4])

    # reward
    reward = csv_list[5]

    # drawing figures
    draw_circles(good_circles_ndarray, bad_circles_ndarray, selected_good_circles, reward,  path, fname,ep,step)

    return None

def convert_str_ndarray(str_list):
    """Input: strings which saved in trans.csv (good_circles or bad circles)

    output: nd_array
    """
    str_list = str_list.replace("[", "")
    str_list = str_list.replace("]", "")
    str_list = np.array([s.split() for s in str_list.splitlines()]) 
    str_list = str_list.astype(np.float)

    return str_list

def draw_circles(good_circles_xyr, bad_circles_xyr, selected_good_circles, reward, path, fname, ep, step):
    """Draw the selected circles into different color and lines
    if the input is total good_xyr datas, bad_circles datas, selected good circles data
    """
    
    # plt.figure()
    fig, ax = plt.subplots()
    # plt.axes().set_aspect('equal')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))

    # draw selected good circles
    selected_good_circles = np.array(selected_good_circles)
    
    # draw good circles
    for good_circle in good_circles_xyr:
        good_circle_xy = tuple(good_circle[:-1])
        good_circle_r = good_circle[-1]

        # create circles
        circle_object_good = plt.Circle(xy=good_circle_xy, radius = max(0.01, good_circle_r), edgecolor = 'g',facecolor ='none', ls = '-', lw = 0.1)

        ax.add_artist(circle_object_good)

    if FLAGS.N_INVARIANT:
        for bad_circle in bad_circles_xyr:
            bad_circle_xy = tuple(bad_circle[:-1])
            bad_circle_r = bad_circle[-1]

            # create circles
            circle_object_bad = plt.Circle(xy=bad_circle_xy, radius = max(0.011, bad_circle_r), edgecolor = 'r',facecolor ='none', ls = ':', lw = 1.0)

            ax.add_artist(circle_object_bad)

    for selected_good_circle in selected_good_circles:
       
        selected_good_circle_xyr = good_circles_xyr[selected_good_circle]
        selected_good_circle_xy = tuple(selected_good_circle_xyr[:-1])
        selected_good_circle_r = selected_good_circle_xyr[-1]
  
        # create circles
        circle_object_selected = plt.Circle(xy=selected_good_circle_xy, radius = max(0.01, selected_good_circle_r), edgecolor = 'b',facecolor ='none', ls = '-', lw = 0.8)

        ax.add_artist(circle_object_selected)

    # plt.xlabel('Step')
    # plt.ylabel('circles')
    # plt.legend()

    plt.suptitle(
        'ep_'+ str(ep) +'_'+ 'step_' + str(step) + '_reward_' + str(reward)
    )

    if not os.path.exists(path + 'trans_fig/'):
        os.makedirs(path + 'trans_fig/')

    fig.savefig(path + 'trans_fig/'+fname+'.png')

    return None



if __name__ == '__main__':
    folder_name = 'greedy-NONE_Seed-0_Eq-20_In-5_K-4_Ep-1_Step-5000_0826-023611'
    path = './circle_env/train-result/'+folder_name + '/'
    file_name = 'transitions'

    # episode and steps to draw
    episodes = [0]
    steps = [0,1,2,3,4,5,6,7,8,9, 2000,2001, 2002, 2003,2004, 4505]

    
    # # setting the time
    # now = time.localtime()
    # s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


    csv_list = read_csv(path+ file_name+'.csv')

    for episode in episodes:
        for step in steps:
            # print('ep,step', episode, step)
            file_name_ep_step = file_name + 'ep_' + str(episode) +'step_' + str(step)
            save_trans_circles(csv_list[50*episode+int(step/500)*10 + step%500], path, file_name_ep_step,episode,step)
            print('h')