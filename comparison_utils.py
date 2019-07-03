# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:43:49 2019

@author: Asun
"""

import os
import pdb
from M2SVM_Optimal_Weights.optimization_problem_utils.system import sys
import numpy as np
import M2SVM_Optimal_Weights.optimization_problem_utils.error_handling as error
from statistics import mean 

def run_optimization_problem_given_solver(solver,
                                          problem,
                                          neos_flag,
                                          number_of_variables,
                                          number_of_constraints,
                                          sense_opt_problem,
                                          maximum_number_iterations_multistart,
                                          folder_results,
                                          csv_file_name_multistart):
    
    if problem == "m2svm_optimal_weights":
        run_m2svm_optimal_weights(solver = solver,
                                  problem = problem,
                                  neos_flag = neos_flag,
                                  number_of_variables = number_of_variables,
                                  number_of_constraints = number_of_constraints,
                                  sense_opt_problem = sense_opt_problem,
                                  maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                  folder_results = folder_results,
                                  csv_file_name_multistart = csv_file_name_multistart)
    else:
        raise error.my_custom_error("The given optimization problem does not exist❌🚫 Please, check 👁‍🗨 that the name is well-written ✍")
    
    return "The optimization problem has been solved"

def run_m2svm_optimal_weights(solver,
                              problem,
                              neos_flag,
                              number_of_variables,
                              number_of_constraints,
                              sense_opt_problem,
                              maximum_number_iterations_multistart,
                              folder_results,
                              csv_file_name_multistart):
    
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
    
    output = sys2.learn_line(method = method,
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
                             maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                             folder_results = folder_results)
    write_results_m2svm_optimal_weights(output = output,
                                        solver = solver,
                                        problem = problem,
                                        neos_flag = neos_flag,
                                        number_of_variables = number_of_variables,
                                        number_of_constraints = number_of_constraints,
                                        sense_opt_problem = sense_opt_problem,
                                        maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                        folder_results = folder_results,
                                        csv_file_name_multistart = csv_file_name_multistart)
    return output

def write_results_m2svm_optimal_weights(output,
                                        solver,
                                        problem,
                                        neos_flag,
                                        number_of_variables,
                                        number_of_constraints,
                                        sense_opt_problem,
                                        maximum_number_iterations_multistart,
                                        folder_results,
                                        csv_file_name_multistart):
    
    folder_results_msvm = folder_results
    if neos_flag:
        neos_string = 'yes'
    else:
        neos_string = 'no'
    csv_file = folder_results_msvm + csv_file_name_multistart + '_' + solver + '_neos_' + neos_string +'.csv'
    file_to_write = open(csv_file, 'w+')
    file_to_write.write('multistart' + ',' + 'objective value' + ','+ 'elapsed time\n')
    
    file_to_write = open(csv_file, 'a')
    objective_values = []
    elapsed_times = []
    line = 1 #Do not change this parameter, since it depends on the data structure of the optimization algorithm.
    for iteration_multistart in range(1, maximum_number_iterations_multistart + 1):    
        objective_values.append(output[line]['results_multistart'][iteration_multistart - 1]['objective_value'])
        elapsed_times.append(output[line]['results_multistart'][iteration_multistart - 1]['elapsed_time'])
        file_to_write.write(str(iteration_multistart) + ',' + "{:.3e}".format(objective_values[iteration_multistart - 1]) + ','+ "{:.3e}".format(elapsed_times[iteration_multistart - 1]) +'\n')
    file_to_write.close()
            
    mean_objective_values = mean(objective_values)
    mean_elapsed_times = mean(elapsed_times)
    maximum_objective_value = max(objective_values)
    minimum_objective_value = min(objective_values)
    maximum_elapsed_time = max(elapsed_times)
    minimum_elapsed_time = min(elapsed_times)
    
    
    
    csv_file_summary_results = folder_results_msvm + 'summary_results.csv'
    file_to_write_summary = open(csv_file_summary_results, 'a+')
    file_to_write_summary.write(problem + ','  + neos_string + ','+ solver + ',' + str(number_of_variables) + ','+ str(number_of_constraints) + ','+ sense_opt_problem + ','+  "{:.3e}".format(mean_objective_values) + ','+ "{:.3e}".format(maximum_objective_value)+ ',' + "{:.3e}".format(minimum_objective_value) + ','+ "{:.3e}".format(mean_elapsed_times) + ','+ "{:.3e}".format(maximum_elapsed_time) + ','+ "{:.3e}".format(minimum_elapsed_time) + '\n')
    file_to_write_summary.close()
        
    return 0


def create_folder_results_if_it_doesnt_exits(folder_results):
    if not (os.path.isdir('./' + folder_results)):
            os.mkdir(folder_results)
    return 0

def initialize_summary_results_file(folder_results,
                                    csv_file_summary_results):
    csv_file_summary_results = folder_results + csv_file_summary_results + '.csv'
    file_to_write_summary = open(csv_file_summary_results, 'a+')
    file_to_write_summary.write('problem, ' + 'neos, '      + 'solver, '+ '# variables, '            + '# constraints, '            + 'sense, '           + 'mean obj. val., '            + 'max obj. val., '              + 'min obj. val., '              + 'mean comp. time, '       + 'max comp. time, '          + 'min comp. time, '          + '\n')
    
    return 0
    
    