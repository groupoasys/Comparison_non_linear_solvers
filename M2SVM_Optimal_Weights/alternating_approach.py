# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:39:15 2019

@author: Asun
"""
import numpy as np
import first_step_alternating_approach as fsap
import math
import pdb
import pandas as pd
import second_step_alternating_approach as ssap
import error_handling as error

def alternating_approach(maximum_number_iterations_alternating_approach,
                         threshold_difference_objective_values_second_step,
                         default_difference_objective_values_second_step,
                         seed_random_prediction_values,
                         index_SVM_regularization_parameter,
                         sample_names,
                         lowest_label_value,
                         highest_label_value,
                         transformed_label_all_samples,
                         SVM_regularization_parameter,
                         seed_initialize_parameters,
                         number_of_nodes,
                         bounds_weights,
                         new_label_values,
                         data_all_samples,
                         number_of_renewable_energy,
                         correspondence_time_period_line_all_samples,
                         sample_by_line,
                         maximum_number_iterations_multistart,
                         perturbation_multistart_variables,
                         seed_multistart,
                         default_new_objective_value_second_step,
                         line,
                         initial_weights,
                         label_values):
    
    iteration_alternating_approach = 0
    difference_objective_values = default_difference_objective_values_second_step
    variables_previous_iteration = np.array([math.nan]*number_of_nodes)
    new_objective_value_second_step = default_new_objective_value_second_step
    output_alternating_approach_by_iteration = {}
    while iteration_alternating_approach <= maximum_number_iterations_alternating_approach or default_difference_objective_values_second_step <= threshold_difference_objective_values_second_step:
        
        initial_variables = get_initial_variables(iteration_alternating_approach = iteration_alternating_approach,
                                                  seed_initialize_parameters = seed_initialize_parameters,
                                                  number_of_nodes = number_of_nodes,
                                                  index_SVM_regularization_parameter = index_SVM_regularization_parameter,
                                                  bounds_weights = bounds_weights,
                                                  variables_previous_iteration = variables_previous_iteration,
                                                  initial_weights = initial_weights,
                                                  line = line)
        ###################################################################################################################################################################################################
        # ASUN: The alpha variables are the optimal solution of the Multiclass Support Vector Machine optimization problem. Such a problem is a convex quadratic problem which is solved using Cplex.
        # Since the objective of this repo is to compare the performance of the different non-linear solvers, it is desirable to avoid the computation of any extra optimization problem. Hence,I copy 
        # and fix the results of the alpha variables obtained for a toy example with 12 individuals and SVM_regularization_parameter = 1e-3, i.e., C = 1/(1e-3) = 1e3. However, the code which computes
        # the optimal variables in a general case will be just commented and not deleted.
        # If further information about this point is necessary, please do not hesitate to contact Asun =)
        
#        output_first_step = fsap.run_first_step_alternating_approach(SVM_regularization_parameter = SVM_regularization_parameter,
#                                                                     number_of_nodes = number_of_nodes,
#                                                                     index_SVM_regularization_parameter = index_SVM_regularization_parameter,
#                                                                     sample_to_train = sample_names[0],
#                                                                     transformed_label_all_samples = transformed_label_all_samples,
#                                                                     new_label_values = new_label_values,
#                                                                     data_all_samples = data_all_samples,
#                                                                     number_of_renewable_energy = number_of_renewable_energy,
#                                                                     correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
#                                                                     sample_by_line = sample_by_line,
#                                                                     initial_variables = initial_variables,
#                                                                     sample_names = sample_names,
#                                                                     line = line,
#                                                                     label_values = label_values)
#        alpha_variables = output_first_step['alpha_variables']
        ###################################################################################################################################################################################################
        output_first_step = {}
        output_first_step['accuracy'] = {}
        output_first_step['accuracy'][sample_names[1]] = 1e2
        alpha_variables = pd.DataFrame(data = {1: [1e-6, 1e-7, 0.999],
                                              2: [1e-6, 1e-7, 0.999],
                                              3: [1e-5, 1e-7, 0.999],
                                              4: [0.000112, 0.203, 0.796],
                                              5: [1e-5, 0.999, 1e-7],
                                              6: [1e-5, 1e-6, 0.999],
                                              7: [0.00108, 0.796, 0.203],
                                              8: [1e-5, 0.999, 1e-7],
                                              9: [1e-5, 0.999, 1e-7],
                                              10: [1e-6, 0.999, 1e-7],
                                              11: [1e-6, 0.999, 1e-7],
                                              12: [1e-6, 0.999, 1e-7]},
                                        index = new_label_values)
        if(len(data_all_samples['training_1']) != alpha_variables.shape[1] or SVM_regularization_parameter != 1e-3):
            raise error.my_custom_error("The data set or the regularization parameter has changed. Please, ask Asun to check the status of the alpha variables")
        
        pdb.set_trace()
        output_second_step = ssap.run_second_step_alternating_approach(alpha_variables = alpha_variables,
                                                                       initial_variables = initial_variables,
                                                                       maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                                       new_label_values = new_label_values,
                                                                       transformed_label_all_samples = transformed_label_all_samples,
                                                                       sample_alpha_variables = sample_names[0],
                                                                       sample_to_train = sample_names[1],
                                                                       number_of_nodes = number_of_nodes,
                                                                       bounds_weights = bounds_weights,
                                                                       perturbation_multistart_variables = perturbation_multistart_variables,
                                                                       seed_multistart = seed_multistart,
                                                                       index_SVM_regularization_parameter = index_SVM_regularization_parameter,
                                                                       iteration_alternating_approach = iteration_alternating_approach,
                                                                       sample_names = sample_names,
                                                                       SVM_regularization_parameter = SVM_regularization_parameter,
                                                                       data_all_samples = data_all_samples,
                                                                       correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                       sample_by_line = sample_by_line,
                                                                       number_of_renewable_energy = number_of_renewable_energy,
                                                                       line = line)
        
        if (output_first_step['accuracy'][sample_names[1]] > output_second_step['best_accuracy'][sample_names[1]]):
           variables_previous_iteration = initial_variables
        else:
           variables_previous_iteration = output_second_step['optimal_weights']
        
        
        old_objective_value_second_step = new_objective_value_second_step
        new_objective_value_second_step = output_second_step['optimal_objective_value']
        difference_objective_values = new_objective_value_second_step - old_objective_value_second_step
        last_iteration_alternating_approach = iteration_alternating_approach
        iteration_alternating_approach = iteration_alternating_approach + 1
        
        output_alternating_approach_by_iteration[iteration_alternating_approach] = {'first_step': output_first_step,
                                                                                    'second_step': output_second_step}
    
    output_alternating_approach = output_alternating_approach_by_iteration[last_iteration_alternating_approach]
    output_alternating_approach['last_iteration_alternating_approach'] = last_iteration_alternating_approach
    return output_alternating_approach

def get_initial_variables(iteration_alternating_approach,
                          seed_initialize_parameters,
                          number_of_nodes,
                          index_SVM_regularization_parameter,
                          bounds_weights,
                          variables_previous_iteration,
                          initial_weights,
                          line):
    np.random.seed(seed = seed_initialize_parameters + line + index_SVM_regularization_parameter + iteration_alternating_approach)
    
    if(iteration_alternating_approach == 0):
        if (initial_weights[0] == None):
            initial_weights = pd.DataFrame(data = np.random.uniform(low = bounds_weights['lower_bound'],
                                                                    high = bounds_weights['upper_bound_initial_solution'],
                                                                    size = number_of_nodes))
        else:
            initial_weights = pd.DataFrame(data = initial_weights)
    else:
        weights_previous_iteration = variables_previous_iteration
        initial_weights = weights_previous_iteration
        
    initial_variables = initial_weights
            
    return initial_variables