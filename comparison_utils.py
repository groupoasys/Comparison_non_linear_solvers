# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:43:49 2019

@author: Asun
"""

from M2SVM_Optimal_Weights.optimization_problem_utils.system import sys
import M2SVM_Optimal_Weights.optimization_problem_utils.error_handling as error

import os
import pdb
import numpy as np
from statistics import mean
import optimization_problem_utils.my_project as mp
from optimization_problem_utils import run_mymodel
import logging
import pandas as pd
import time

def run_optimization_problem_given_solver(solver,
                                          problem,
                                          neos_flag,
                                          number_of_variables,
                                          number_of_constraints,
                                          sense_opt_problem,
                                          maximum_number_iterations_multistart,
                                          folder_results,
                                          csv_file_name_multistart,
                                          ampl_flag):
    
    if problem == "m2svm_optimal_weights":
        run_m2svm_optimal_weights(solver = solver,
                                  problem = problem,
                                  neos_flag = neos_flag,
                                  number_of_variables = number_of_variables,
                                  number_of_constraints = number_of_constraints,
                                  sense_opt_problem = sense_opt_problem,
                                  maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                  folder_results = folder_results,
                                  csv_file_name_multistart = csv_file_name_multistart,
                                  ampl_flag = ampl_flag)
    elif problem == "inverse_optimization_related_problem":
        run_inverse_optimization_related_problem(solver = solver,
                                                 problem = problem,
                                                 neos_flag = neos_flag,
                                                 number_of_variables = number_of_variables,
                                                 number_of_constraints = number_of_constraints,
                                                 sense_opt_problem = sense_opt_problem,
                                                 maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                 folder_results = folder_results,
                                                 csv_file_name_multistart = csv_file_name_multistart,
                                                 ampl_flag = ampl_flag)
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
                              csv_file_name_multistart,
                              ampl_flag):
    
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
                             folder_results = folder_results,
                             ampl_flag = ampl_flag)
    write_results_m2svm_optimal_weights(output = output,
                                        solver = solver,
                                        problem = problem,
                                        neos_flag = neos_flag,
                                        number_of_variables = number_of_variables,
                                        number_of_constraints = number_of_constraints,
                                        sense_opt_problem = sense_opt_problem,
                                        maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                        folder_results = folder_results,
                                        csv_file_name_multistart = csv_file_name_multistart,
                                        ampl_flag = ampl_flag)
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
                                        csv_file_name_multistart,
                                        ampl_flag):
    
    folder_results_msvm = folder_results
    if neos_flag:
        neos_string = 'yes'
    else:
        neos_string = 'no'
    if ampl_flag:
        ampl_string = 'yes'
    else:
        ampl_string = 'no'
    csv_file = folder_results_msvm + csv_file_name_multistart + '_' + solver + '_neos_' + neos_string + '_ampl_' + ampl_string + '.csv'
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
    file_to_write_summary.write(problem + ','  + neos_string + ',' + ampl_string + ','+ solver + ',' + str(number_of_variables) + ','+ str(number_of_constraints) + ','+ sense_opt_problem + ','+  "{:.3e}".format(mean_objective_values) + ','+ "{:.3e}".format(maximum_objective_value)+ ',' + "{:.3e}".format(minimum_objective_value) + ','+ "{:.3e}".format(mean_elapsed_times) + ','+ "{:.3e}".format(maximum_elapsed_time) + ','+ "{:.3e}".format(minimum_elapsed_time) + '\n')
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
    file_to_write_summary.write('problem, ' + 'neos, ' + 'ampl, '     + 'solver, '+ '# variables, '            + '# constraints, '            + 'sense, '           + 'mean obj. val., '            + 'max obj. val., '              + 'min obj. val., '              + 'mean comp. time, '       + 'max comp. time, '          + 'min comp. time, '          + '\n')
    
    return 0

def run_inverse_optimization_related_problem(solver,
                                             problem,
                                             neos_flag,
                                             number_of_variables,
                                             number_of_constraints,
                                             sense_opt_problem,
                                             maximum_number_iterations_multistart,
                                             folder_results,
                                             csv_file_name_multistart,
                                             ampl_flag):
    
    # Please, do not change this function here
    def run_mymodel(config, results_dir='./results/', solver='cplex'):
        """ Run my model
        
        Parameters:
            config (str): configuration file
            results_dir (str): directory to store results
            solver (str): solver to use (CPLEX, glpk, etc.)
    
        """
        results_dir = os.path.join(results_dir,config['output_files']['folder'])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        out_keys = ['info', 'theta', 'u', 'x', 'y']
    
        dict_out = {}
        for i in range(len(out_keys)):
            dict_out[out_keys[i]] = pd.DataFrame([]) 
    
        if config['model_cfg']['how_to_read_data']=='endogenously':
            logging.info("## Creating input data endogenously ##")
            data = {}
            data['parameters'] = pd.DataFrame([], index=['theta', 'n'], columns=['Value'])
            data['parameters'].loc['theta', 'Value'] = config['model_cfg']['theta'] 
            data['parameters'].loc['n', 'Value'] = config['model_cfg']['n_sample']
            data['parameters'].index.name = 'p'
        
            logging.info("## Generating input data for the parameter u ##")
            np.random.seed(seed = 1133)
            data['u'] = pd.DataFrame(np.random.uniform(-2,1,data['parameters'].loc['n', 'Value']), index=range(1, int(data['parameters'].loc['n', 'Value'])+1), columns=['u'])
            data['u'].index.name = 'i'
            
        elif config['model_cfg']['how_to_read_data']=='exogenously':
            # Reading the data
            logging.info("## Reading input data exogenously ##")
            data = mp.parse_excel(config['input_files']['data'], config) 
                 
        logging.info("## Solving model 1 to generate x ##")
        start_time = time.time()
        
        # Create instance model
        instance_mymodel = mp.model1(data, config)
            
        solved_instance, solver_status, solver_solutions  = mp.run_solver(instance_mymodel, config['solver_cfg'])
            
        print("--- %s seconds ---" % (time.time() - start_time))
          
        if solver_status.termination_condition.index == 8:  # Termination condition of solver: Optimal
            dict_out['theta'].loc['theta', 'Value'] = data['parameters'].loc['theta', 'Value']
            dict_out['x'].loc[:, 'x'] = mp.pyomo_to_pandas(solved_instance, 'x').iloc[:,0]
            dict_out['y'].loc[:, 'y'] = mp.pyomo_to_pandas(solved_instance, 'y').iloc[:,0]
            dict_out['u'].loc[:, 'u'] = data['u'].iloc[:, 0]
            dict_out['info'].loc['Time', 'Value'] = (time.time() - start_time)
    
        mp.dict_pandas_to_excel(dict_out, dir=results_dir, filename=config['output_files']['filename'])

    def main():
        config = mp.read_yaml('./optimization_problem_utils/configs/config_model1.yml')
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%d/%m/%y %H:%M:%S',
                            filename=config['log_file'],
                            level=logging.DEBUG)
    
        run_mymodel(config,
                    results_dir = folder_results,
                    solver = solver)
    
    main()
        
    return 0
    
    