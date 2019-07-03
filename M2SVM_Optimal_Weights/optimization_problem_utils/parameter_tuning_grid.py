# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:05:16 2019

@author: Asun
"""

import pdb
import numpy as np
import M2SVM_Optimal_Weights.optimization_problem_utils.alternating_approach as alt

def tune_parameters_grid(SVM_regularization_parameter_grid,
                         sample_by_line,
                         label,
                         line,
                         sample_names,
                         seed_random_prediction_values,
                         lowest_label_value,
                         highest_label_value,
                         sample_to_get_best_parameters,
                         maximum_number_iterations_alternating_approach,
                         threshold_difference_objective_values_second_step,
                         default_difference_objective_values_second_step,
                         seed_initialize_parameters,
                         number_of_nodes,
                         bounds_weights,
                         label_values,
                         new_label_values,
                         data,
                         number_of_renewable_energy,
                         correspondence_time_period_line_all_samples,
                         maximum_number_iterations_multistart,
                         perturbation_multistart_variables,
                         seed_multistart,
                         default_new_objective_value_second_step,
                         initial_weights):
    performance_results_all_SVM_regularization_parameters = get_performance_results_all_SVM_regularization_parameters(SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,
                                                                                                                      sample_by_line = sample_by_line,
                                                                                                                      label = label,
                                                                                                                      line = line,
                                                                                                                      sample_names = sample_names,
                                                                                                                      seed_random_prediction_values = seed_random_prediction_values,
                                                                                                                      lowest_label_value = lowest_label_value,
                                                                                                                      highest_label_value = highest_label_value,
                                                                                                                      maximum_number_iterations_alternating_approach = maximum_number_iterations_alternating_approach,
                                                                                                                      threshold_difference_objective_values_second_step = threshold_difference_objective_values_second_step,
                                                                                                                      default_difference_objective_values_second_step = default_difference_objective_values_second_step,
                                                                                                                      seed_initialize_parameters = seed_initialize_parameters,
                                                                                                                      number_of_nodes = number_of_nodes,
                                                                                                                      bounds_weights = bounds_weights,
                                                                                                                      label_values = label_values,
                                                                                                                      new_label_values = new_label_values,
                                                                                                                      data = data,
                                                                                                                      number_of_renewable_energy = number_of_renewable_energy,
                                                                                                                      correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                                                                      maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                                                                                      perturbation_multistart_variables = perturbation_multistart_variables,
                                                                                                                      seed_multistart = seed_multistart,
                                                                                                                      default_new_objective_value_second_step = default_new_objective_value_second_step,
                                                                                                                      initial_weights = initial_weights)
    best_results_tune_parameters_grid = get_best_results_tune_parameters_grid(performance_results_all_SVM_regularization_parameters = performance_results_all_SVM_regularization_parameters,
                                                                              sample_to_get_best_parameters = sample_to_get_best_parameters,
                                                                              SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,)

    return best_results_tune_parameters_grid


def get_accuracy_all_samples(sample_names,
                             label_all_samples,
                             prediction_all_samples):
    accuracy_all_samples = {}
    for sample_name in sample_names:
        accuracy_all_samples[sample_name] = get_accuracy_by_sample(label = label_all_samples[sample_name].values,
                                                                   prediction = prediction_all_samples[sample_name].values)
    return accuracy_all_samples

def get_accuracy_by_sample(label,
                           prediction):
    number_good_classified = np.sum(label == prediction)
    accuracy = (number_good_classified/len(prediction))*100
    
    return accuracy

def get_label_all_samples(sample_names,
                          label,
                          line,
                          sample_by_line):
    label_all_samples = {}
    for sample_name in sample_names:
        label_all_samples[sample_name] = label.loc[sample_by_line[sample_name], label.columns[line]]

    return label_all_samples

def get_data_all_samples(sample_names,
                         data,
                         sample_by_line):
    data_all_samples = {}
    for sample_name in sample_names:
        data_all_samples[sample_name] = data.loc[sample_by_line[sample_name],:]
    return data_all_samples

def get_prediction_all_samples(sample_names,
                               line,
                               sample_by_line,
                               transformed_label_all_samples,
                               seed_random_prediction_values,
                               lowest_label_value,
                               highest_label_value,
                               index_SVM_regularization_parameter,
                               maximum_number_iterations_alternating_approach,
                               threshold_difference_objective_values_second_step,
                               default_difference_objective_values_second_step,
                               SVM_regularization_parameter,
                               seed_initialize_parameters,
                               number_of_nodes,
                               bounds_weights,
                               new_label_values,
                               data_all_samples,
                               number_of_renewable_energy,
                               correspondence_time_period_line_all_samples,
                               maximum_number_iterations_multistart,
                               perturbation_multistart_variables,
                               seed_multistart,
                               default_new_objective_value_second_step,
                               initial_weights,
                               label_values):
    
    output_alternating_approach = alt.alternating_approach(maximum_number_iterations_alternating_approach = maximum_number_iterations_alternating_approach,
                                                           threshold_difference_objective_values_second_step = threshold_difference_objective_values_second_step,
                                                           default_difference_objective_values_second_step = default_difference_objective_values_second_step,
                                                           seed_random_prediction_values = seed_random_prediction_values,
                                                           index_SVM_regularization_parameter = index_SVM_regularization_parameter,
                                                           sample_names = sample_names,
                                                           lowest_label_value = lowest_label_value,
                                                           highest_label_value = highest_label_value,
                                                           transformed_label_all_samples = transformed_label_all_samples,
                                                           SVM_regularization_parameter = SVM_regularization_parameter,
                                                           seed_initialize_parameters = seed_initialize_parameters,
                                                           number_of_nodes = number_of_nodes,
                                                           bounds_weights = bounds_weights,
                                                           new_label_values = new_label_values,
                                                           data_all_samples = data_all_samples,
                                                           number_of_renewable_energy = number_of_renewable_energy,
                                                           correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                           sample_by_line = sample_by_line,
                                                           maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                           perturbation_multistart_variables = perturbation_multistart_variables,
                                                           seed_multistart = seed_multistart,
                                                           default_new_objective_value_second_step = default_new_objective_value_second_step,
                                                           line = line,
                                                           initial_weights = initial_weights,
                                                           label_values = label_values)
    prediction_all_samples = output_alternating_approach['second_step']['best_prediction']
    output_prediction_all_samples = {'prediction': prediction_all_samples,
                                     'weights': output_alternating_approach['second_step']['optimal_weights'],
                                     'results_multistart':output_alternating_approach['second_step']['results_multistart']}
    
    return output_prediction_all_samples
    
    
def get_performance_results_all_SVM_regularization_parameters(SVM_regularization_parameter_grid,
                                                              sample_by_line,
                                                              label,
                                                              line,
                                                              sample_names,
                                                              seed_random_prediction_values,
                                                              lowest_label_value,
                                                              highest_label_value,
                                                              maximum_number_iterations_alternating_approach,
                                                              threshold_difference_objective_values_second_step,
                                                              default_difference_objective_values_second_step,
                                                              seed_initialize_parameters,
                                                              number_of_nodes,
                                                              bounds_weights,
                                                              label_values,
                                                              new_label_values,
                                                              data,
                                                              number_of_renewable_energy,
                                                              correspondence_time_period_line_all_samples,
                                                              maximum_number_iterations_multistart,
                                                              perturbation_multistart_variables,
                                                              seed_multistart,
                                                              default_new_objective_value_second_step,
                                                              initial_weights):
    performance_results_all_SVM_regularization_parameters = []
    for SVM_regularization_parameter in SVM_regularization_parameter_grid:
        index_SVM_regularization_parameter = SVM_regularization_parameter_grid.index(SVM_regularization_parameter)
        performance_results_by_SVM_regularization_parameter = {}
        performance_results_by_SVM_regularization_parameter["SVM_regularization_parameter"] = SVM_regularization_parameter
        data_all_samples = get_data_all_samples(sample_names = sample_names,
                                                data = data,
                                                sample_by_line = sample_by_line)
        performance_results_by_SVM_regularization_parameter['data'] = data_all_samples
        label_all_samples = get_label_all_samples(sample_names = sample_names,
                                                  label = label,
                                                  line = line,
                                                  sample_by_line = sample_by_line)
        performance_results_by_SVM_regularization_parameter["label"] = label_all_samples
        transformed_label_all_samples = get_transformed_label_all_samples(label_all_samples = label_all_samples,
                                                                          label_values = label_values,
                                                                          new_label_values = new_label_values)
        
        performance_results_by_SVM_regularization_parameter["transformed_label"] = transformed_label_all_samples
        output_prediction_all_samples = get_prediction_all_samples(sample_names = sample_names,
                                                                    line = line,
                                                                    sample_by_line = sample_by_line,
                                                                    transformed_label_all_samples = transformed_label_all_samples,
                                                                    seed_random_prediction_values = seed_random_prediction_values,
                                                                    lowest_label_value = lowest_label_value,
                                                                    highest_label_value = highest_label_value,
                                                                    index_SVM_regularization_parameter = index_SVM_regularization_parameter,
                                                                    maximum_number_iterations_alternating_approach = maximum_number_iterations_alternating_approach,
                                                                    threshold_difference_objective_values_second_step = threshold_difference_objective_values_second_step,
                                                                    default_difference_objective_values_second_step = default_difference_objective_values_second_step,
                                                                    SVM_regularization_parameter = SVM_regularization_parameter,
                                                                    seed_initialize_parameters = seed_initialize_parameters,
                                                                    number_of_nodes = number_of_nodes,
                                                                    bounds_weights = bounds_weights,
                                                                    new_label_values = new_label_values,
                                                                    data_all_samples = data_all_samples,
                                                                    number_of_renewable_energy = number_of_renewable_energy,
                                                                    correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                    maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                                    perturbation_multistart_variables = perturbation_multistart_variables,
                                                                    seed_multistart = seed_multistart,
                                                                    default_new_objective_value_second_step = default_new_objective_value_second_step,
                                                                    initial_weights = initial_weights,
                                                                    label_values = label_values)
        prediction_all_samples = output_prediction_all_samples['prediction']
        performance_results_by_SVM_regularization_parameter["prediction"] = prediction_all_samples
        transformed_predictions_all_samples = get_transformed_prediction_all_samples(prediction_all_samples = prediction_all_samples,
                                                                                     label_values = label_values,
                                                                                     new_label_values = new_label_values)
        performance_results_by_SVM_regularization_parameter["transformed_predictions"] = transformed_predictions_all_samples
        accuracy_all_samples = get_accuracy_all_samples(label_all_samples = label_all_samples,
                                                        prediction_all_samples = transformed_predictions_all_samples,
                                                        sample_names = sample_names)
        performance_results_by_SVM_regularization_parameter['weights'] = output_prediction_all_samples['weights']
        performance_results_by_SVM_regularization_parameter["accuracy"] = accuracy_all_samples
        performance_results_by_SVM_regularization_parameter['results_multistart'] = output_prediction_all_samples['results_multistart']
        
        performance_results_all_SVM_regularization_parameters.append(performance_results_by_SVM_regularization_parameter)
    return performance_results_all_SVM_regularization_parameters

def get_best_results_tune_parameters_grid(performance_results_all_SVM_regularization_parameters,
                                          sample_to_get_best_parameters,
                                          SVM_regularization_parameter_grid):
    
    all_accuracies_sample_get_best_parameters = [performance_results_by_SVM_regularization_parameter['accuracy'][sample_to_get_best_parameters] for performance_results_by_SVM_regularization_parameter in performance_results_all_SVM_regularization_parameters]
    best_accuracy = max(all_accuracies_sample_get_best_parameters)
    best_SVM_regularization_parameter_index = np.argmax(all_accuracies_sample_get_best_parameters)
    best_SVM_regularization_parameter = SVM_regularization_parameter_grid[best_SVM_regularization_parameter_index]
    accuracy_all_samples = performance_results_all_SVM_regularization_parameters[best_SVM_regularization_parameter_index]['accuracy']
    prediction_all_samples = performance_results_all_SVM_regularization_parameters[best_SVM_regularization_parameter_index]['transformed_predictions']
    weights = performance_results_all_SVM_regularization_parameters[best_SVM_regularization_parameter_index]['weights']
    best_results = {'best_SVM_regularization_parameter': best_SVM_regularization_parameter,
                    'best_accuracy': best_accuracy,
                    'accuracy_all_samples': accuracy_all_samples,
                    'prediction_all_samples': prediction_all_samples,
                    'weights': weights,
                    'results_multistart': performance_results_all_SVM_regularization_parameters[best_SVM_regularization_parameter_index]['results_multistart']}
    return best_results

def get_transformed_label_all_samples(label_all_samples,
                                      label_values,
                                      new_label_values):
    transformed_label_all_samples = label_all_samples.copy()
    for key in transformed_label_all_samples.keys():
        transformed_label_all_samples[key] = transformed_label_all_samples[key].replace(to_replace = label_values,
                                                                                        value = new_label_values)
        
    return transformed_label_all_samples

def get_transformed_prediction_all_samples(prediction_all_samples,
                                      label_values,
                                      new_label_values):
    transformed_prediction_all_samples = prediction_all_samples.copy()
    for key in transformed_prediction_all_samples.keys():
        transformed_prediction_all_samples[key] = transformed_prediction_all_samples[key].replace(to_replace = new_label_values,
                                                                                                  value = label_values)
        
    return transformed_prediction_all_samples

    