# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:43:49 2019

@author: Asun
"""

from optimization_problem_utils import *
import os
from optimization_problem_utils.system import sys
import numpy as np
import pdb
import optimization_problem_utils.error_handling as error

def run_optimization_problem_given_solver(solver,
                                          problem,
                                          neos_flag,
                                          number_of_variables,
                                          number_of_constraints,
                                          sense_opt_problem):
    
    if problem == "m2svm_optimal_weights":
        run_m2svm_optimal_weights(solver = solver,
                                  problem = problem,
                                  neos_flag = neos_flag,
                                  number_of_variables = number_of_variables,
                                  number_of_constraints = number_of_constraints,
                                  sense_opt_problem = sense_opt_problem)
    else:
        raise error.my_custom_error("The given optimization problem does not exist❌🚫 Please, check 👁‍🗨 that the name is well-written ✍")
    
    return "The optimization problem has been solved"

def run_m2svm_optimal_weights(solver,
                              problem,
                              neos_flag,
                              number_of_variables,
                              number_of_constraints,
                              sense_opt_problem):
    
    # Please, do not change this function here
    main_directory_data = 'optimization_problem_utils/'
    directory_data = 'msvm_toy_example'
    source_data_file = 'demand_values.csv'
    generators_data_file = 'generators.csv'
    lines_data_file = 'lines.csv'
    
    data_file_path = os.path.join(main_directory_data, directory_data, source_data_file)
    generators_file_path = os.path.join(main_directory_data, directory_data, generators_data_file)
    lines_file_path = os.path.join(main_directory_data, directory_data, lines_data_file)
    load_shedding = 1000
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
    
    sys2 = sys(gen_file = generators_file_path,
               lin_file = lines_file_path,
               data_file = data_file_path,
               c_shed = load_shedding,
               slack_bus = slack_bus)
    
    SVM_regularization_parameter_grid = [10**range_element for range_element in range(-3, -2)]
    
    
    sys2.learning_test_data(ini_train=ini_train,
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
                    SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,
                    solver = solver,
                    problem = problem,
                    neos_flag = neos_flag,
                    number_of_variables = number_of_variables,
                    number_of_constraints = number_of_constraints,
                    sense_opt_problem = sense_opt_problem,
                    main_directory_data = main_directory_data)
    
    return 0
    
    