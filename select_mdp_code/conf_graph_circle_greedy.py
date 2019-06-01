from __future__ import division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import argparse
from matplotlib.pyplot import figure
# figure(num=None, figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')
# Constants
z = 1.96  # 95% confidence
plt.rcParams["figure.figsize"] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20 


def extract_data(filename):
    data = np.loadtxt(open(filename, 'r'), delimiter=',', skiprows=1)
    return data[:, 2]

def calculate(agent, ne, k, directory, datalen, command):
    seed_list = [0, 1, 2, 3]
    data = []

    if args.k == 1:
        inv = 1
        ep=500
    if args.k == 6:
        inv = 0
        ep=1400

    for i in seed_list:
        filename = '{}/run_{}_Eq-{}_In-{}_K-{}_Ep-{}_Step-2500_Command-{}_Lr-0.001_Buf-50000_Bat-32_NwrkEx-6_Layers-4_Xav-1_seed_{}-tag-reward.csv'.format(
            directory, agent, ne, inv, k, ep, command, i)
        print('filename',filename)
        data.append(extract_data(filename)[:datalen])

    data = np.array(data, dtype=np.float32)
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    tmp = z * std / np.sqrt(seed_list.__len__())
    lb = avg - tmp
    gap = 2 * tmp
    ub = avg + tmp
    return [avg, lb, gap, ub]

def process(args):

    # Initialize variables
    directory = '0108_Circle_k{}'.format(args.k)
    if args.k == 1:
        ne_list = [5, 20, 50]
        agent_list = ['Vanilla', 'Sorting',  'I-sharing']
        agent_dict = {
            agent_list[0] : 'dqn-VANILLA',
            agent_list[1] : 'dqn-VANILLA-sort',
            agent_list[2] : 'dqn-SHAREEINET'
        }
        datalen = 100  #TODO
    elif args.k == 6:
        ne_list = [50, 200]
        agent_list = ['I-Sharing', 'U-Sharing', "P-Sharing", "P-Sharing (1)"]
        agent_dict = {
            agent_list[0] : 'dqn-DIFFEINET',
            agent_list[1] : 'dqn-SHAREEINET',
            agent_list[2] : 'dqn-PROGRESSIVE',
            agent_list[3] : 'dqn-PROGRESSIVE'
        }
        datalen = 100  #TODO
    if args.k == -1:
        pass
    else:
        raise Exception("Wrong input.")

    # Constants for drawing
      #TODO
    if args.k == 1:
        x = np.arange(1, datalen+1) 
        x = 5 * x
    if args.k == 6:
        x = np.arange(1, datalen+1) 
        x = 14 *x

    if args.k==1 or args.k==6:
        for i, ne in enumerate(ne_list):
            print("Figure {} - N = {}".format(i+1, ne))
            plt.figure()

            if args.k==1:
                command = 1
            if args.k ==6:
                command = 5


            top_ub = 0
            for agent in agent_list:
                if agent == "P-Sharing (1)":
                    command = 1
                conf_data = calculate(agent_dict[agent], ne, args.k, directory, datalen, command)
                avg = conf_data[0]
                lb  = conf_data[1]
                gap = conf_data[2]
                ub  = conf_data[3]

                assert datalen == len(avg)

                plt.plot(x, avg, label = agent, linewidth=2.5)
                
                plt.fill_between(xx, lb, ub, alpha=0.2)

                plt.xlabel('Training episode', fontsize=30)
                plt.ylabel('Average reward', fontsize=30)
                # plt.title('Number of selectable elements: {}'.format(ne), fontsize=25)
                plt.grid(True, linestyle='dashed')
                plt.legend(prop={'size':18})
                if args.k == 1:
                    tick_scale = 0.01
                if args.k ==6:
                    tick_scale = 0.05
                top_ub = max(top_ub, max(ub))
                plt.yticks(np.arange((int(int(1/tick_scale)*min(lb)))*tick_scale, (int(int(1/tick_scale)*top_ub)+1)*tick_scale, tick_scale))
                plt.tight_layout()
                plt.savefig("{}/circle_K-{}_conf-graph_ne-{}.pdf".format(directory, args.k, ne))
                plt.savefig("{}/circle_K-{}_conf-graph_ne-{}.png".format(directory, args.k, ne))

                # Save data
                csv_data = zip(*[x, avg, lb, gap, ub])
                fname = '{}/conf-data_circle_K-{}_Ne-{}_datalen-{}.csv'.format(directory, args.k, ne, datalen)
                i=0
                with open(fname, 'w') as f:
                    csv.writer(f).writerows(csv_data)

                # draw greedy heuristic
            if args.k == 6:
                if ne == 50:
                    greedy_avg = np.ones(100) *0.072
                if ne == 200:
                    greedy_avg = np.ones(100) *0.077
                plt.plot(x, greedy_avg, label = "Heuristic", linewidth = 1.5, linestyle ='dashdot')

                plt.legend(prop={'size':18})
                plt.tight_layout()
                plt.savefig("{}/circle_K-{}_conf-graph_ne-{}.pdf".format(directory, args.k, ne))
                plt.savefig("{}/circle_K-{}_conf-graph_ne-{}.png".format(directory, args.k, ne))

    if args.k == -1:
        plt.figure()

        x = np.arange(15) * 20
        gred = [0.22712929, 0.19343377, 0.19915247, 0.19810776, 0.16242123, 0.09264948,
0.10167533, 0.09415358, 0.09351535, 0.07551487, 0.07149908, 0.07760623,
0.08295525, 0.06218723, 0.06464102]

        Input_List = [50, 200]
        Output_List = [50, 200]
        SEED_List = [0,1,2,3]
        RELOAD_List = [66,67,68,69,70]
        # avg_data_early = np.zeros(15)
        # avg_data = 0
        for input_ in Input_List:
            for output_ in Output_List:
                avg_data = 0
                if input_ == 200 and output_ ==200:
                    avg_data_early = np.zeros(15)
                for seed_ in SEED_List:
                    for reload_ in RELOAD_List:
                        path = './circle_env_good/csv_result/'+str(input_) + '_' +str(output_)+'_' + str(seed_) + '_' + str(reload_) +'.csv'
                        # print(path)
                        data = np.loadtxt(open(path, 'r'), delimiter=',')
                        # try:
                        data = np.reshape(data, [-1,125])
                        # except:
                        #     print(data)
                        #     print('input,output,seed, reload', input_, output_, seed_, reload_)
                        data = np.average(data, 0)
                        if input_ == 200 and output_ ==200:
                            avg_data_early += data[:15]
                        avg_data += np.average(data)
                        
                if input_ == 200 and output_ ==200:
                    avg_data_early = avg_data_early / float(len(SEED_List) * len(RELOAD_List))
                avg_data = avg_data / float(len(SEED_List) * len(RELOAD_List))
                print('input',input_,' output', output_, 'avg', avg_data)


        plt.plot(x, gred, label = 'Heuristic', linewidth=2.5)
        plt.plot(x, avg_data_early, label = 'P-Sharing', linewidth=2.5)

        # plt.title('Number of selectable elements: {}'.format(ne), fontsize=25)
        plt.grid(True, linestyle='dashed')
        tick_scale = 0.05
        plt.yticks(np.arange(0,7)*tick_scale)


        plt.legend(prop={'size':18})
        plt.xlabel('Test step', fontsize=30)
        plt.ylabel('Average reward', fontsize=30)
        plt.tight_layout()

        plt.savefig("circle_env_good/csv_result/"  + "HeurvsPsharing.png")
        plt.savefig("circle_env_good/csv_result/" +"HeurvsPsharing.pdf")
        print(avg_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, required=True)     # 1 | 3

    args = parser.parse_args()

    process(args)
