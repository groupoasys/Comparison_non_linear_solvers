# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:36:32 2019

@author: Asun
"""

import pyomo.environ as pe
import pdb
import numpy as np
import pandas as pd

def multiclass_SVM(weights,
                   SVM_regularization_parameter,
                   label_to_train,
                   new_label_values,
                   data_to_train,
                   number_of_renewable_energy,
                   correspondence_time_period_line_all_samples,
                   sample_to_train,
                   sample_by_line):
    
    model = pe.ConcreteModel('multiclass_SVM')
    
    model.label_to_train = label_to_train
    model.data_to_train = data_to_train
    model.SVM_regularization_parameter = SVM_regularization_parameter
    model.weights = pd.concat([weights]*(1 + number_of_renewable_energy))
    model.correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples
    model.sample_to_train = sample_to_train
    model.sample_by_line = sample_by_line
    
    
    number_of_labels = max(new_label_values)
    number_of_time_periods = len(label_to_train)
    
    model.indexes_labels = pe.RangeSet(number_of_labels)
    model.indexes_time_periods = pe.RangeSet(number_of_time_periods)
    
    model.alpha_variables = pe.Var(model.indexes_labels,
                                   model.indexes_time_periods,
                                   within = pe.NonNegativeReals)
    
    model.constraint_sum_variables_equal_one = pe.Constraint(model.indexes_time_periods,
                                                             rule = constraint_sum_variables_equal_one)
    
    model.objective_function = pe.Objective(rule = objective_function,
                                            sense = pe.minimize)
    
    return model

def objective_function(model):
    objective_value = sum(sum(model.alpha_variables[label, time_period]*dirac_delta(value_1 = label,
                                                                                    value_2 = model.label_to_train.iat[time_period - 1, 0])
                              for time_period in model.indexes_time_periods)
                          for label in model.indexes_labels) + (1/(4*model.SVM_regularization_parameter))*sum(sum(sum(((model.alpha_variables[label, time_period_1] - dirac_delta(value_1 = label,
                                                                                                                                                                                 value_2 = model.label_to_train.iat[time_period_1 - 1, 0]))*(model.alpha_variables[label, time_period_2] - dirac_delta(value_1 = label,
                                                                                                                                                                                                                                                                                                      value_2 = model.label_to_train.iat[time_period_2 - 1, 0]))*(kernel_function(model = model, time_period_1=time_period_1, time_period_2= time_period_2, weights= model.weights))) 
                                                                                                                      for time_period_2 in model.indexes_time_periods)
                                                                                                                  for time_period_1 in model.indexes_time_periods) 
                                                                                                               for label in model.indexes_labels)
    return objective_value

def constraint_sum_variables_equal_one(model,
                                       time_period):
    constraint = sum(model.alpha_variables[label, time_period] for label in model.indexes_labels) == 1
    return constraint


def dirac_delta(value_1,
                value_2):
    if(value_1 == value_2):
        dirac_delta_value = 1
    else:
        dirac_delta_value = 0
    return dirac_delta_value

def kernel_function(model,
                    time_period_1,
                    time_period_2,
                    weights = pd.DataFrame(data = np.nan, index = [0], columns = [0])):
    individual_1 = model.data_to_train.iloc[time_period_1 - 1, :]
    individual_2 = model.data_to_train.iloc[time_period_2 - 1, :]
    correspondence_time_period_line_all_samples = model.correspondence_time_period_line_all_samples
    original_index_time_period_1 = model.sample_by_line[model.sample_to_train][time_period_1 - 1]
    original_index_time_period_2 = model.sample_by_line[model.sample_to_train][time_period_2 - 1]
    correspondence_time_period_line_sample_to_train = correspondence_time_period_line_all_samples[model.sample_to_train]
    
    line_associated_to_time_period_1 = correspondence_time_period_line_sample_to_train.loc[correspondence_time_period_line_sample_to_train['time_period'] == original_index_time_period_1,'line'].item()
    line_associated_to_time_period_2 = correspondence_time_period_line_sample_to_train.loc[correspondence_time_period_line_sample_to_train['time_period'] == original_index_time_period_2,'line'].item()
    
    squared_difference_between_individuals = pd.DataFrame(data = (individual_1 - individual_2)**2)
    if pd.isnull(weights).all().values[0]:
        values_weights_times_squared_difference_between_individuals = []
        for node in model.indexes_nodes:
            values_weights_times_squared_difference_between_individuals.append(model.weights_variables[node]*squared_difference_between_individuals.iat[node - 1, 0])
        for node in model.indexes_nodes:
            values_weights_times_squared_difference_between_individuals.append(model.weights_variables[node]*squared_difference_between_individuals.iat[model.number_of_nodes + node - 1, 0])
        weights_times_squared_difference_between_individuals = pd.DataFrame(data = values_weights_times_squared_difference_between_individuals)
    else:
        weights_times_squared_difference_between_individuals = pd.DataFrame(data = weights.values*squared_difference_between_individuals.values)
    kernel_value = (1 + dirac_delta(value_1 = line_associated_to_time_period_1,
                                    value_2 = line_associated_to_time_period_2))*pe.exp(-weights_times_squared_difference_between_individuals.sum().item())
    
    return kernel_value







