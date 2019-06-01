from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import csv

# Constants
z = 1.96  # 95% confidence
rule_based_avg = [0.14148697, 0.14376807, 0.1360485, 0.12614523]

def extract_data(filename):
    data = np.loadtxt(open(filename, 'r'), delimiter=',', skiprows=1)
    return data[:, 2]

def calculate(agent, ne, directory):
    seed_list = [0, 1, 2, 3]
    data = []
    for i in seed_list:
        filename = '{}/run_{}_Eq-{}_In-5_K-2_Ep-3000_Step-500_Lr-0.003_Buf-100000_Bat-128_NwrkEx-6_Layers-4_Xav-1_seed_{}-tag-reward.csv'.format(
            directory, agent, ne, i)
        data.append(extract_data(filename))

    data = np.array(data, dtype=np.float32)
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    tmp = z * std / np.sqrt(seed_list.__len__())
    lb = avg - tmp
    gap = 2 * tmp
    ub = avg + tmp
    return [avg, lb, gap, ub]

def process():

    # Initialize variables
    directory = 'predator-prey/k2'
    ne_list = [20, 50, 100, 200]
    agent_list = ['Inter-Parameter Sharing']
    agent_dict = {
        agent_list[0] : 'dqn-SHAREEINET'
    }

    # Constants for drawing
    datalen = 150
    x = np.arange(1, datalen+1) * 10000 / 500

    for i, ne in enumerate(ne_list):
        print("Figure {} - N = {}".format(i+1, ne))
        plt.figure()

        for agent in agent_list:
            conf_data = calculate(agent_dict[agent], ne, directory)
            avg = conf_data[0]/rule_based_avg[i]*100.
            lb  = conf_data[1]/rule_based_avg[i]*100.
            gap = conf_data[2]/rule_based_avg[i]*100.
            ub  = conf_data[3]/rule_based_avg[i]*100.

            assert datalen == len(avg)

            plt.plot(x, avg, label = agent)
            plt.fill_between(x, lb, ub, alpha=0.4)

            plt.xlabel('Training episode', fontsize=18)
            plt.ylabel('Relative reward (%)', fontsize=18)
            plt.title('Number of selectable elements: {}'.format(ne), fontsize=20)
            plt.grid(True, linestyle='dashed')
            plt.legend(prop={'size':14})
            plt.savefig("{}/pp_K-2_conf-graph_ne-{}.pdf".format(directory, ne))
            plt.savefig("{}/pp_K-2_conf-graph_ne-{}.png".format(directory, ne))

            # Save data
            csv_data = zip(*[x, avg, lb, gap, ub])
            fname = '{}/conf-data_pp_K-2_Ne-{}_datalen-{}.csv'.format(directory, ne, datalen)
            with open(fname, 'w') as f:
                csv.writer(f).writerows(csv_data)

if __name__ == '__main__':

    process()
