# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:27:57 2019

@author: Asun
"""
import numpy as np
import pdb
import multiclass_svm as msvm
from pyomo.opt import SolverFactory
import pandas as pd
import parameter_tuning_grid as ptg
import prediction as pred
import timeit
import pyomo.environ as pe

def run_first_step_alternating_approach(SVM_regularization_parameter,
                                        number_of_nodes,
                                        index_SVM_regularization_parameter,
                                        sample_to_train,
                                        transformed_label_all_samples,
                                        new_label_values,
                                        data_all_samples,
                                        number_of_renewable_energy,
                                        correspondence_time_period_line_all_samples,
                                        sample_by_line,
                                        initial_variables,
                                        sample_names,
                                        line,
                                        label_values):
    if not type(initial_variables) == 'pandas.core.frame.DataFrame':
        weights = pd.DataFrame(data = initial_variables)
    else:
        weights = initial_variables
    label_to_train = transformed_label_all_samples[sample_to_train].to_frame()
    label_to_train.index = range(1, len(label_to_train) + 1)
    
    data_to_train = data_all_samples[sample_to_train]
#    solver = pe.SolverManagerFactory("neos")
    solver = pe.SolverFactory("cplex")
    #solver.options['optimality_target'] = 3
    SVM_model = msvm.multiclass_SVM(weights = weights,
                                    SVM_regularization_parameter = SVM_regularization_parameter,
                                    label_to_train = label_to_train,
                                    new_label_values = new_label_values,
                                    data_to_train = data_to_train,
                                    number_of_renewable_energy = number_of_renewable_energy,
                                    correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                    sample_to_train = sample_to_train,
                                    sample_by_line = sample_by_line)
    initial_time_first_step = timeit.default_timer()
#    results_solver = solver.solve(SVM_model,
#                                  tee = True,
#                                  opt = 'cplex')
    results_solver = solver.solve(SVM_model,
                                  tee = True)
    final_time_first_step = timeit.default_timer()
    elapsed_time_first_step = final_time_first_step - initial_time_first_step
    results_SVM = get_results_first_step_alternating_approach(SVM_model = SVM_model,
                                                              sample_names = sample_names,
                                                              transformed_label_all_samples = transformed_label_all_samples,
                                                              new_label_values = new_label_values,
                                                              weights_variables = weights,
                                                              data_all_samples = data_all_samples,
                                                              sample_by_line = sample_by_line,
                                                              line = line,
                                                              number_of_renewable_energy = number_of_renewable_energy,
                                                              correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                              SVM_regularization_parameter = SVM_regularization_parameter,
                                                              elapsed_time = elapsed_time_first_step,
                                                              label_values = label_values)
    return results_SVM


def get_results_first_step_alternating_approach(SVM_model,
                                                sample_names,
                                                transformed_label_all_samples,
                                                new_label_values,
                                                weights_variables,
                                                data_all_samples,
                                                sample_by_line,
                                                line,
                                                number_of_renewable_energy,
                                                correspondence_time_period_line_all_samples,
                                                SVM_regularization_parameter,
                                                elapsed_time,
                                                label_values):
    objective_value = SVM_model.objective_function.expr()
    alpha_variables = pd.DataFrame(index = SVM_model.indexes_labels,
                                   columns = SVM_model.indexes_time_periods)
    for label in SVM_model.indexes_labels:
        for time_period in SVM_model.indexes_time_periods:
            alpha_variables.iloc[label - 1, time_period - 1] = SVM_model.alpha_variables[label, time_period].value
    
    predictions = {}
    true_label = {}
    accuracy = {}
    for sample_name in sample_names:
        true_label[sample_name] = transformed_label_all_samples[sample_name]
        predictions[sample_name] = pred.get_prediction_by_sample(alpha_variables = alpha_variables,
                                                                 SVM_regularization_parameter = SVM_regularization_parameter,
                                                                 weights_variables = weights_variables,
                                                                 first_sample = sample_names[0],
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
    transformed_predictions_all_samples = ptg.get_transformed_prediction_all_samples(prediction_all_samples = predictions,
                                                                                     label_values = label_values,
                                                                                     new_label_values = new_label_values)
    results_SVM = {'objective_value': objective_value,
                   'alpha_variables': alpha_variables,
                   'label': true_label,
                   'predictions': predictions,
                   'transformed_predictions': transformed_predictions_all_samples,
                   'accuracy': accuracy,
                   'elapsed_time': elapsed_time}
    return results_SVM
    