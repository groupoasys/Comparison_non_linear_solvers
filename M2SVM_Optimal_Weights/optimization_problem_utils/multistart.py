# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:29:33 2019

@author: Asun
"""
import numpy as np
import pdb
import M2SVM_Optimal_Weights.optimization_problem_utils.parameter_tuning_grid as ptg
import pandas as pd
import M2SVM_Optimal_Weights.optimization_problem_utils.optimization_problem_second_step as opss
import pyomo.environ as pe
import M2SVM_Optimal_Weights.optimization_problem_utils.prediction as pred
import timeit

def run_multistart_approach(alpha_variables,
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
                            line,
                            solver):
    
    results_multistart = {}
    solver_name = solver
    for iteration_multistart in range(maximum_number_iterations_multistart):
        results_multistart[iteration_multistart] = {}
        
        initial_variables_multistart = get_initial_variables_multistart(iteration_multistart = iteration_multistart,
                                                                        initial_variables = initial_variables,
                                                                        number_of_nodes = number_of_nodes,
                                                                        bounds_weights = bounds_weights,
                                                                        perturbation_multistart_variables = perturbation_multistart_variables,
                                                                        seed_multistart = seed_multistart,
                                                                        index_SVM_regularization_parameter = index_SVM_regularization_parameter,
                                                                        iteration_alternating_approach = iteration_alternating_approach)
        
        solver = pe.SolverManagerFactory("neos")
        label_to_train = (transformed_label_all_samples[sample_to_train]).to_frame()
        data_to_train = data_all_samples[sample_to_train]
        label_alpha_variables = (transformed_label_all_samples[sample_alpha_variables].to_frame())
        data_alpha_variables = data_all_samples[sample_alpha_variables]
        multistart_model = opss.optimization_problem_second_step(alpha_variables = alpha_variables,
                                                                 number_of_nodes = number_of_nodes,
                                                                 new_label_values = new_label_values,
                                                                 label_to_train = label_to_train,
                                                                 SVM_regularization_parameter = SVM_regularization_parameter,
                                                                 data_to_train = data_to_train,
                                                                 correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                 sample_by_line = sample_by_line,
                                                                 sample_to_train = sample_to_train,
                                                                 sample_alpha_variables = sample_alpha_variables,
                                                                 number_of_renewable_energy = number_of_renewable_energy,
                                                                 initial_variables_multistart = initial_variables_multistart,
                                                                 label_alpha_variables = label_alpha_variables,
                                                                 data_alpha_variables = data_alpha_variables)
            
        
        initial_time_second_step = timeit.default_timer()
        
        results_solver = solver.solve(multistart_model,
                                      tee = True,
                                      opt = solver_name)
        
        final_time_second_step = timeit.default_timer()
        elapsed_time_second_step = final_time_second_step - initial_time_second_step
        objective_value_multistart = multistart_model.objective_function.expr()
        epigraph_variables = pd.DataFrame(index = multistart_model.indexes_time_periods)
        weights_variables = pd.DataFrame(index = multistart_model.indexes_nodes)
        for time_period in multistart_model.indexes_time_periods:
            epigraph_variables.loc[time_period, 0] = multistart_model.epigraph_variables[time_period].value
        
        for node in multistart_model.indexes_nodes:
            weights_variables.loc[node,0] = multistart_model.weights_variables[node].value
        
        predictions = {}
        true_label = {}
        accuracy = {}
        for sample_name in sample_names:
            true_label[sample_name] = transformed_label_all_samples[sample_name]
            predictions[sample_name] = pred.get_prediction_by_sample(alpha_variables = alpha_variables,
                                                                     SVM_regularization_parameter = SVM_regularization_parameter,
                                                                     weights_variables = weights_variables,
                                                                     first_sample = sample_to_train,
                                                                     second_sample = sample_name,
                                                                     transformed_label_all_samples = transformed_label_all_samples,
                                                                     data_all_samples = data_all_samples,
                                                                     correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                     sample_by_line = sample_by_line,
                                                                     new_label_values = new_label_values,
                                                                     line = line,
                                                                     number_of_renewable_energy = number_of_renewable_energy)
            accuracy[sample_name] = ptg.get_accuracy_by_sample(label = true_label[sample_name].values,
                                                               prediction = predictions[sample_name].values)
            
        results_multistart[iteration_multistart]['weights'] = weights_variables
        results_multistart[iteration_multistart]['epigraph_variables'] = epigraph_variables
        results_multistart[iteration_multistart]['prediction'] = predictions
        results_multistart[iteration_multistart]['true_label'] = true_label
        results_multistart[iteration_multistart]['accuracy'] = accuracy
        results_multistart[iteration_multistart]['objective_value'] = objective_value_multistart
        results_multistart[iteration_multistart]['elapsed_time'] = elapsed_time_second_step
    return results_multistart


def get_initial_variables_multistart(iteration_multistart,
                                     initial_variables,
                                     number_of_nodes,
                                     bounds_weights,
                                     perturbation_multistart_variables,
                                     seed_multistart,
                                     index_SVM_regularization_parameter,
                                     iteration_alternating_approach):
    
    np.random.seed(seed = seed_multistart + index_SVM_regularization_parameter + iteration_alternating_approach + iteration_multistart)
    initial_weights = initial_variables
    if (iteration_multistart == 0):
        initial_weights_multistart = initial_weights
        
    else:
        initial_weights_multistart = pd.DataFrame(data = np.random.uniform(low = np.maximum(np.array(initial_weights.iloc[:, 0] - perturbation_multistart_variables['weights']),
                                                                                            np.repeat(bounds_weights['lower_bound'], number_of_nodes)),
                                                                           high = np.array(initial_weights.iloc[:, 0] + perturbation_multistart_variables['weights']),
                                                                           size = number_of_nodes))

    initial_variables_multistart = initial_weights_multistart

    return initial_variables_multistart










