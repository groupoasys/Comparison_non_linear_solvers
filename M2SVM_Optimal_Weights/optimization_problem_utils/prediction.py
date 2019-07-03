# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:46:48 2019

@author: Asun
"""
import pdb
import M2SVM_Optimal_Weights.optimization_problem_utils.multiclass_svm as msvm
import numpy as np
import pandas as pd

def get_prediction_by_sample(alpha_variables,
                             SVM_regularization_parameter,
                             weights_variables,
                             first_sample,
                             second_sample,
                             transformed_label_all_samples,
                             data_all_samples,
                             correspondence_time_period_line_all_samples,
                             sample_by_line,
                             new_label_values,
                             line,
                             number_of_renewable_energy):
    
    label_first_sample = transformed_label_all_samples[first_sample]
    label_second_sample = transformed_label_all_samples[second_sample]
    
    time_range_first_sample = np.arange(len(label_first_sample))
    
    kernel_matrix = get_kernel_matrix(data_all_samples = data_all_samples,
                                      weights = weights_variables,
                                      correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                      sample_by_line = sample_by_line,
                                      first_sample = first_sample,
                                      second_sample = second_sample,
                                      line = line,
                                      number_of_renewable_energy = number_of_renewable_energy,
                                      label_first_sample = label_first_sample,
                                      label_second_sample = label_second_sample)
    delta_minus_alpha = get_delta_minus_alpha_vector(new_label_values = new_label_values,
                                                     label_first_sample = label_first_sample,
                                                     alpha_variables = alpha_variables,
                                                     time_range_first_sample = time_range_first_sample)
    
    score_all_time_periods_and_labels = float((1/(2*SVM_regularization_parameter)))*np.dot(delta_minus_alpha, kernel_matrix)
    predictions = 1 + np.argmax(score_all_time_periods_and_labels, axis = 0)
    predictions = pd.Series(data = predictions, index = range(1, len(label_second_sample) + 1))
    return predictions


def get_kernel_matrix(data_all_samples,
                      weights,
                      correspondence_time_period_line_all_samples,
                      sample_by_line,
                      first_sample,
                      second_sample,
                      line,
                      number_of_renewable_energy,
                      label_first_sample,
                      label_second_sample):
    original_time_indexes_first_sample = sample_by_line[first_sample]
    original_time_indexes_second_sample = sample_by_line[second_sample]
    
    correspondence_time_period_line_first_sample = correspondence_time_period_line_all_samples[first_sample]
    correspondence_time_period_line_second_sample = correspondence_time_period_line_all_samples[second_sample]
    
    lines_associated_to_time_indexes_first_sample = np.array(correspondence_time_period_line_first_sample[correspondence_time_period_line_first_sample.time_period.isin(original_time_indexes_first_sample)]['line'], dtype = int)
    if (lines_associated_to_time_indexes_first_sample[0]<0):
        lines_associated_to_time_indexes_first_sample = np.repeat(a = line + 1,
                                                                  repeats = len(label_first_sample))
    lines_associated_to_time_indexes_second_sample = np.array(correspondence_time_period_line_second_sample[correspondence_time_period_line_second_sample.time_period.isin(original_time_indexes_second_sample)]['line'], dtype = int)
    if (lines_associated_to_time_indexes_second_sample[0]<0):
        lines_associated_to_time_indexes_second_sample = np.repeat(a = line + 1,
                                                                   repeats = len(label_second_sample))
    one_plus_delta_matrix = 1 + (lines_associated_to_time_indexes_first_sample[:, None] == lines_associated_to_time_indexes_second_sample[None,:]).astype(int)
    
    data_first_sample_multiplied_by_weights = np.multiply(np.array(data_all_samples[first_sample]), np.tile(np.sqrt(np.array(weights.loc[:,0])), 1 + number_of_renewable_energy))
    data_second_sample_multiplied_by_weights = np.multiply(np.array(data_all_samples[second_sample]), np.tile(np.sqrt(np.array(weights.loc[:,0])), 1 + number_of_renewable_energy))
    
    #Here, we apply that (a - b)^2 = a^2 + b^2 - 2ab. The first two terms applies the Einstein sum, and the third one apply the usual dot product.
    weighted_sum_squared_difference_individuals = np.einsum('ij,ij->i', 
                                                            data_first_sample_multiplied_by_weights,
                                                            data_first_sample_multiplied_by_weights)[:,None] + np.einsum('ij,ij->i',
                                                                                                                          data_second_sample_multiplied_by_weights, 
                                                                                                                          data_second_sample_multiplied_by_weights) - 2*np.dot(data_first_sample_multiplied_by_weights, 
                                                                                                                                                                               data_second_sample_multiplied_by_weights.T)
    
    exponential_matrix = np.exp(-weighted_sum_squared_difference_individuals)
    kernel_matrix = np.multiply(one_plus_delta_matrix, exponential_matrix)
    
    return kernel_matrix


def get_delta_minus_alpha_vector(new_label_values,
                                 label_first_sample,
                                 alpha_variables,
                                 time_range_first_sample):
    delta_minus_alpha = np.empty(shape = (len(new_label_values), len(label_first_sample)), dtype = float)
    for label in new_label_values:
        delta_minus_alpha_lambda_function = lambda time_period, label = label, label_first_sample = label_first_sample, alpha_variables = alpha_variables: msvm.dirac_delta(value_1 = label,
                                                                                                                                                                            value_2 = label_first_sample.iloc[time_period]) - alpha_variables.loc[label, time_period + 1]
            
        delta_minus_alpha[label - 1,:] = np.array(list(map(delta_minus_alpha_lambda_function, time_range_first_sample)))
    
    return(delta_minus_alpha)
    
