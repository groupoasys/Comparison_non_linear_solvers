# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:53:25 2019

@author: Asun
"""

from __future__ import division

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import os
from system import sys
import pandas as pd
import numpy as np
import pdb
import pickle


directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)

directory_data = 'msvm_toy_example'
source_data_file = 'demand_values.csv'
generators_data_file = 'generators.csv'
lines_data_file = 'lines.csv'

data_file_path = os.path.join(directory_data, source_data_file)
generators_file_path = os.path.join(directory_data, generators_data_file)
lines_file_path = os.path.join(directory_data, lines_data_file)
load_shedding = 1000
number_of_nodes = 3
net_demand = True
ini_train = 0
end_train = 32
ini_test = 15
end_test = 16
weight_ptdf = False
method = 'illustrative_m2svm_optimization'
level = -1
slack_bus = 1
weights_values = np.abs([[1, 1, 1],[1, 1, 1],[1, 1, 1]], dtype = float)


full_data = pd.read_csv(data_file_path,
                        index_col = False)
generators_data = pd.read_csv(generators_file_path)

sys2 = sys(gen_file = generators_file_path,
           lin_file = lines_file_path,
           data_file = data_file_path,
           c_shed = load_shedding,
           slack_bus = slack_bus)

weights_choices = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [0, 1, 1],
                         [0.5, 0.5, 1],
                         [0.5, 0, 1],
                         [0, 0.5, 1],
                         [5, 5, 1],
                         [5, 0, 1],
                         [0, 5, 1]],
                         dtype = float)
SVM_regularization_parameter_grid = [10**range_element for range_element in range(-3, -2)]

#for choice in range(len(weights_choices)):
#    weights_values[1,:] = weights_choices[choice,:]
all_data_information = sys2.learning_test_data(ini_train=ini_train,
                                                ini_test=ini_test,
                                                end_train=end_train,
                                                end_test=end_test,
                                                weight_ptdf=weight_ptdf,
                                                net_demand=net_demand)
sys2.learn_line(method = method,
                    level = level,
                    net_demand = net_demand,
                    weight_ptdf = weight_ptdf,
                    weights_values = weights_values,
                    SVM_regularization_parameter_grid = SVM_regularization_parameter_grid)
#if weight_ptdf:
#    sys2.solve_uc_illustrative_example('results_msvm/illustrative_UC_results_weights_ptdf.csv')
#else:
#    sys2.solve_uc_illustrative_example('results_msvm/illustrative_UC_results_weights_1.csv')
#    
#
#file_to_load = 'results_msvm/illustrative_all_results_random.pydata'
#file_to_read = open(file_to_load, 'rb')
#results_to_save = pickle.load(file_to_read)




