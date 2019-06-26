# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:10:54 2019

@author: Asun
"""

import pyomo.environ as pe
import multiclass_svm as msvm
import pdb
import pandas as pd
import numpy as np
import logging
import error_handling as error

def optimization_problem_second_step(alpha_variables,
                                     number_of_nodes,
                                     new_label_values,
                                     label_to_train,
                                     SVM_regularization_parameter,
                                     data_to_train,
                                     correspondence_time_period_line_all_samples,
                                     sample_by_line,
                                     sample_to_train,
                                     sample_alpha_variables,
                                     number_of_renewable_energy,
                                     initial_variables_multistart,
                                     label_alpha_variables,
                                     data_alpha_variables):
    model = pe.ConcreteModel('optimization_problem_second_step')
    model.label_to_train = label_to_train
    model.SVM_regularization_parameter = SVM_regularization_parameter
    model.data_to_train = data_to_train
    model.correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples
    model.sample_by_line = sample_by_line
    model.sample_to_train = sample_to_train
    model.sample_alpha_variables = sample_alpha_variables
    model.number_of_renewable_energy = number_of_renewable_energy
    model.number_of_nodes = number_of_nodes
    model.label_alpha_variables = label_alpha_variables
    model.data_alpha_variables = data_alpha_variables
    
    
    number_of_labels = max(new_label_values)
    number_of_time_periods = len(label_to_train)
    number_of_time_periods_first_sample = alpha_variables.shape[1]
    
    model.alpha_variables = alpha_variables
    
    model.indexes_nodes = pe.RangeSet(model.number_of_nodes)
    model.indexes_time_periods = pe.RangeSet(number_of_time_periods)
    model.indexes_labels = pe.RangeSet(number_of_labels)
    model.indexes_time_periods_first_sample = pe.RangeSet(number_of_time_periods_first_sample)
    
    initial_variables_multistart.index = model.indexes_nodes
    
    model.weights_variables = pe.Var(model.indexes_nodes,
                                     within = pe.NonNegativeReals,
                                     initialize = initial_variables_multistart.iloc[:,0].to_dict())
    model.epigraph_variables = pe.Var(model.indexes_time_periods,
                                      within = pe.Reals)
    
    model.constraint_epigraph = pe.Constraint(model.indexes_time_periods,
                                              model.indexes_labels,
                                              rule = constraint_epigraph)
    model.objective_function = pe.Objective(rule = objective_function,
                                            sense = pe.minimize)
    
    return model

    
def objective_function(model):
    
    objective_value = sum(model.epigraph_variables[time_period] for time_period in model.indexes_time_periods)
    
    return objective_value


def constraint_epigraph(model,
                        time_period,
                        label):
    constraint = model.epigraph_variables[time_period]>= (1/(2*model.SVM_regularization_parameter))*sum((msvm.dirac_delta(value_1 = label,
                                                                                                                          value_2 = model.label_alpha_variables.iat[time_period_1 - 1, 0]) 
                                                                                                        - model.alpha_variables.at[label, time_period_1] 
                                                                                                        - 1 
                                                                                                        + model.alpha_variables.at[model.label_to_train.iat[time_period - 1, 0], time_period_1])*kernel_function_second_step(model = model,
                                                                                                                                                                                                                             time_period_1 = time_period,
                                                                                                                                                                                                                             time_period_2 = time_period_1) for time_period_1 in model.indexes_time_periods_first_sample)- msvm.dirac_delta(value_1 = label,
                                                                                                                                                                                                                                                                                                                      value_2 = model.label_to_train.iat[time_period - 1, 0]) + 1

    return constraint

def kernel_function_second_step(model,
                                time_period_1,
                                time_period_2):
    individual_1 = model.data_to_train.iloc[time_period_1 - 1, :]
    individual_2 = model.data_alpha_variables.iloc[time_period_2 - 1, :]
    correspondence_time_period_line_all_samples = model.correspondence_time_period_line_all_samples
    original_index_time_period_1 = model.sample_by_line[model.sample_to_train][time_period_1 - 1]
    original_index_time_period_2 = model.sample_by_line[model.sample_alpha_variables][time_period_2 - 1]
    correspondence_time_period_line_sample_to_train = correspondence_time_period_line_all_samples[model.sample_to_train]
    correspondence_time_period_line_sample_alpha_variables = correspondence_time_period_line_all_samples[model.sample_alpha_variables]
    
    
    squared_difference_between_individuals = pd.DataFrame(data = (individual_1 - individual_2)**2)
    
    line_associated_to_time_period_1 = correspondence_time_period_line_sample_to_train.loc[correspondence_time_period_line_sample_to_train['time_period'] == original_index_time_period_1,'line'].item()
    line_associated_to_time_period_2 = correspondence_time_period_line_sample_alpha_variables.loc[correspondence_time_period_line_sample_alpha_variables['time_period'] == original_index_time_period_2,'line'].item()
    
    if(model.number_of_renewable_energy > 0):
        raise error.my_custom_error("The optimization of the second step is not designed for a number of renewable energy greater than 0. Please, check the value variable in the kernel function.")
        
    
    value = (1 + msvm.dirac_delta(value_1 = line_associated_to_time_period_1,
                                  value_2 = line_associated_to_time_period_2))*pe.exp(-sum(squared_difference_between_individuals.iloc[node - 1, 0]*model.weights_variables[node] for node in model.indexes_nodes))    
    return value

