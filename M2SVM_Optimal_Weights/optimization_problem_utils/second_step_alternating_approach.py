# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:25:07 2019

@author: Asun
"""
import M2SVM_Optimal_Weights.optimization_problem_utils.multistart as ms
import pdb

def run_second_step_alternating_approach(alpha_variables,
                                         initial_variables,
                                         maximum_number_iterations_multistart,
                                         new_label_values,
                                         transformed_label_all_samples,
                                         sample_alpha_variables,
                                         sample_to_train,
                                         number_of_nodes,
                                         bounds_weights,
                                         perturbation_multistart_variables,
                                         seed_multistart,
                                         index_SVM_regularization_parameter,
                                         iteration_alternating_approach,
                                         sample_names,
                                         SVM_regularization_parameter,
                                         data_all_samples,
                                         correspondence_time_period_line_all_samples,
                                         sample_by_line,
                                         number_of_renewable_energy,
                                         line):
    
    results_multistart = ms.run_multistart_approach(alpha_variables = alpha_variables,
                                                    initial_variables = initial_variables,
                                                    maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                    new_label_values = new_label_values,
                                                    transformed_label_all_samples = transformed_label_all_samples,
                                                    sample_alpha_variables = sample_alpha_variables,
                                                    sample_to_train = sample_to_train,
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
    (best_iteration_multistart,
     best_accuracy) = max([(iteration_multistart, results_multistart[iteration_multistart]['accuracy'][sample_to_train]) for iteration_multistart in results_multistart], key = lambda x: x[1])
    
    best_prediction = results_multistart[best_iteration_multistart]['prediction']
    best_accuracy = results_multistart[best_iteration_multistart]['accuracy']
    associated_label = results_multistart[best_iteration_multistart]['true_label']
    optimal_weights = results_multistart[best_iteration_multistart]['weights']
    optimal_objective_value = results_multistart[best_iteration_multistart]['objective_value']
    output_results_multistart = {'best_iteration': best_iteration_multistart,
                                 'best_accuracy': best_accuracy,
                                 'best_prediction': best_prediction,
                                 'label': associated_label,
                                 'optimal_weights': optimal_weights,
                                 'optimal_objective_value': optimal_objective_value,
                                 'results_multistart': results_multistart}
    return output_results_multistart
    