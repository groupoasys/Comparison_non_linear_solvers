# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:38:24 2019

@author: Asun
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prediction as pred
import pdb
import itertools
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import plotly.graph_objs as go
import math

def plot_colors_classification(self,
                               number_of_point_per_axis,
                                output_first_step,
                                line,
                                transformed_label_all_samples,
                                data_all_samples,
                                sample_by_line,
                                correspondence_time_period_line_all_samples,
                                SVM_regularization_parameter,
                                sample_names,
                                new_label_values,
                                number_of_renewable_energy,
                                label_values,
                                approach,
                                folder_results_msvm,
                                dimensions_to_plot,
                                results_SVM):
    bounds_x = [-1, 1]
    bounds_y = [-1, -0.97]
    number_of_points_x, number_of_points_y = (number_of_point_per_axis, number_of_point_per_axis)
    number_of_points = number_of_points_x*number_of_points_y
    grid_x, grid_y = np.meshgrid(np.linspace(bounds_x[0], bounds_x[1], number_of_points_x),
                                 np.linspace(bounds_y[0], bounds_y[1], number_of_points_y))
    grid_x_one_column = pd.DataFrame(grid_x).unstack()
    grid_x_one_column.index = range(1, number_of_points + 1)
    grid_y_one_column = pd.DataFrame(grid_y).unstack()
    grid_y_one_column.index = range(1, number_of_points + 1)
    grid_x_one_column = pd.DataFrame(data = grid_x_one_column.values)
    grid_y_one_column = pd.DataFrame(data = grid_y_one_column.values)
    first_component = pd.DataFrame([data_all_samples['training_1'][2][1]]*number_of_points) #The value can be different from [1]. It also possible to write, for instance, data_all_samples['training_1'][2][3]
    z = pd.concat([grid_x_one_column, grid_y_one_column, first_component],
                  axis = 1)
    number_of_individuals = len(self.x_train) + len(self.x_test)
    z.index = range(number_of_individuals + 1, len(z) + number_of_individuals + 1)
    z.columns = self.x_train.columns
    alpha_variables = output_first_step[line]['alpha_variables']
    weights_variables = pd.DataFrame(self.weights[line])
    sample_name = 'test_plot'
    transformed_label_all_samples[sample_name] = pd.DataFrame(data = [4]*number_of_points,
                                                              index = range((number_of_individuals + 1), number_of_individuals + 1 +number_of_points))
    data_all_samples[sample_name] = z
    correspondence_time_period_line_all_samples[sample_name] = pd.DataFrame(data = list(zip(range((number_of_individuals + 1), number_of_individuals + 1 +number_of_points), np.repeat(-1, number_of_points))),
                                                                            columns = ['time_period', 'line'])
    sample_by_line[sample_name] = list(range((number_of_individuals + 1), number_of_individuals + 1 +number_of_points))
    
    
    ######################
    #Margins
    first_sample = 'training_1'
    second_sample = sample_name
    
    label_first_sample = transformed_label_all_samples[first_sample]
    label_second_sample = transformed_label_all_samples[second_sample]
    
    time_range_first_sample = np.arange(len(label_first_sample))
    
    kernel_matrix = pred.get_kernel_matrix(data_all_samples = data_all_samples,
                                      weights = weights_variables,
                                      correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                      sample_by_line = sample_by_line,
                                      first_sample = first_sample,
                                      second_sample = second_sample,
                                      line = line,
                                      number_of_renewable_energy = number_of_renewable_energy,
                                      label_first_sample = label_first_sample,
                                      label_second_sample = label_second_sample)
    
    delta_minus_alpha = pred.get_delta_minus_alpha_vector(new_label_values = new_label_values,
                                                     label_first_sample = label_first_sample,
                                                     alpha_variables = alpha_variables,
                                                     time_range_first_sample = time_range_first_sample)
    
    score_all_time_periods_and_labels = float((1/(2*SVM_regularization_parameter)))*np.dot(delta_minus_alpha, kernel_matrix)
    
    positive_margin = (score_all_time_periods_and_labels[1,:] - score_all_time_periods_and_labels[2,:])- 1
    hyperplane = (score_all_time_periods_and_labels[1,:] - score_all_time_periods_and_labels[2,:])
    negative_margin = (score_all_time_periods_and_labels[1,:] - score_all_time_periods_and_labels[2,:])+ 1
    
    ######################
    
    
    
    results = pred.get_prediction_by_sample(alpha_variables = alpha_variables,
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
    
    transformed_results = results.replace(to_replace = new_label_values,
                                          value = label_values)
    colors = transformed_results
    for test_individual in range(1, (len(data_all_samples[sample_name][dimensions_to_plot[1]]) + 1)):
        if (positive_margin[test_individual-1] >0 and hyperplane[test_individual-1] > 0):
            colors[test_individual] = 'dodgerblue'
        elif (positive_margin[test_individual-1]<0 and hyperplane[test_individual-1] > 0):
            colors[test_individual] = 'paleturquoise'
        elif (negative_margin[test_individual-1] <0 and hyperplane[test_individual-1] < 0):
            colors[test_individual] = 'red'
        elif (negative_margin[test_individual-1] >0 and hyperplane[test_individual-1] < 0):
            colors[test_individual] = 'mistyrose'
            
    colors_training = transformed_label_all_samples['training_1'].replace(to_replace = new_label_values,
                                                                 value = ['green','dodgerblue', 'red'])
    plt.scatter(data_all_samples[sample_name][dimensions_to_plot[1]],
                data_all_samples[sample_name][dimensions_to_plot[0]],
                c = colors)
    plt.scatter(data_all_samples['training_1'][dimensions_to_plot[1]],
                data_all_samples['training_1'][dimensions_to_plot[0]],
                c = colors_training,
                marker = 'o',
                edgecolors = 'black')
    epsilon = 0.0005
    plt.axis([bounds_y[0]-epsilon, bounds_y[1]+epsilon, bounds_x[0]-epsilon, bounds_x[1]+epsilon])
    
    weight_string_title = get_string_weights_all_buses_title(self = self,
                                                 line = line)
    SVM_reg_parameter_string_title = get_string_SVM_regularization_parameter_title(SVM_regularization_parameter = SVM_regularization_parameter)
    title_plot = get_title_plot(self = self,
                                weight_string = weight_string_title,
                                SVM_reg_parameter_string = SVM_reg_parameter_string_title,
                                approach = approach)
    plt.title(label = title_plot)
    alpha_true_label = alpha_variables.lookup(transformed_label_all_samples['training_1'].values,alpha_variables.columns)
    
    support_vectors_string = get_vectors_strings(alpha_true_label = alpha_true_label,
                                                 alpha_variables = alpha_variables,
                                                 label_first_sample = label_first_sample)
        
    for point in range(len(transformed_label_all_samples['training_1'])):
        plt.text(data_all_samples['training_1'][dimensions_to_plot[1]].iloc[point] + 0.001,
                 data_all_samples['training_1'][dimensions_to_plot[0]].iloc[point] + 0.0003,
                 support_vectors_string[point])
    
    SVM_reg_parameter_string = get_string_SVM_regularization_parameter(SVM_regularization_parameter = SVM_regularization_parameter)
    weight_string = get_string_weights_all_buses(self = self,
                                                 line = line)
    name_fig = get_name_figure(self = self,
                               folder_results_msvm=folder_results_msvm,
                               SVM_reg_parameter_string=SVM_reg_parameter_string,
                               line=line,
                               approach = approach,
                               weight_string = weight_string)
    plt.savefig(name_fig)
    plt.show()
    #pdb.set_trace()
    

def get_string_weights_all_buses(self,
                                 line):
    weight_string = []
    for bus in range(self.n_bus):
        if (math.modf(self.weights[line,bus])[0] == 0):
            weight_string.append(str(int(self.weights[line,bus])))
        else:
             weight_string.append("".join(str(np.round(self.weights[line,bus], 2)).split(".")))
    return weight_string

def get_string_weights_all_buses_title(self,
                                 line):
    weight_string_title = []
    for bus in range(self.n_bus):
        if (math.modf(self.weights[line,bus])[0] == 0):
            weight_string_title.append(str(int(self.weights[line,bus])))
        else:
             weight_string_title.append(str(np.round(self.weights[line,bus], 2)))
    return weight_string_title

def get_name_figure(self,
                 folder_results_msvm,
                 SVM_reg_parameter_string,
                 line,
                 approach,
                 weight_string):
    if approach == 'random':
        name_fig = folder_results_msvm +'grid_predictions_C_' + SVM_reg_parameter_string + '_weights_'
        for bus in range(self.n_bus):
            name_fig = name_fig + weight_string[bus] + '_'
        name_fig = name_fig +'line_'+ str(line + 1) + '.pdf'
    else:
        name_fig = folder_results_msvm +'grid_predictions_C_'+SVM_reg_parameter_string+'_weights_' + approach + '_line_'+ str(line + 1) + '.pdf'  
    return name_fig

def get_string_SVM_regularization_parameter(SVM_regularization_parameter):
    
    C_value = 1/SVM_regularization_parameter
    if (math.modf(C_value)[0] == 0):
        SVM_reg_parameter_string = str(int(C_value))
    else:
       SVM_reg_parameter_string = "".join(str(C_value).split(".")) 
    
    return SVM_reg_parameter_string
    
def get_title_plot(self,
                   weight_string,
                   SVM_reg_parameter_string,
                   approach):
    
    if approach == 'random':
        title_plot = 'C = ' + SVM_reg_parameter_string +', $\omega$ = ('
        for bus in range(self.n_bus - 1):
            title_plot = title_plot + weight_string[bus] + ', '
        title_plot = title_plot + weight_string[self.n_bus - 1] + ')'
    else:
        title_plot = 'C = ' + SVM_reg_parameter_string +', $\omega$ = ' + approach
    return title_plot

def get_string_SVM_regularization_parameter_title(SVM_regularization_parameter):
    
    C_value = 1/SVM_regularization_parameter
    if (math.modf(C_value)[0] == 0):
        SVM_reg_parameter_string_title = str(int(C_value))
    else:
       SVM_reg_parameter_string_title = str(C_value)
    
    return SVM_reg_parameter_string_title

def get_vectors_strings(alpha_true_label,
                        alpha_variables,
                        label_first_sample):
    
    support_vectors_string = ["SV"]*len(alpha_true_label)
    
    indexes_margin = get_indexes_margin_vectors(alpha_true_label = alpha_true_label,
                                                alpha_variables = alpha_variables,
                                                label_first_sample = label_first_sample)
    
    indexes_non_support_vectors = list(itertools.compress(range(1, len(support_vectors_string) + 1), (np.around(alpha_true_label, decimals = 2) == 1).tolist()))
    for index in indexes_non_support_vectors:
        support_vectors_string[index - 1]= "NSV"
    
    for index in indexes_margin:
        support_vectors_string[index - 1]= "??"
        
    return support_vectors_string

def get_indexes_margin_vectors(alpha_true_label,
                               alpha_variables,
                               label_first_sample):
    indexes_margin = []
    indexes_margin += list(map(lambda label: list(filter(lambda individual, label = label, alpha_variables = alpha_variables, label_first_sample = label_first_sample: np.around(alpha_variables.loc[label, individual], decimals = 2) == 1 and label != label_first_sample.loc[individual], alpha_variables.columns)), alpha_variables.index))
    indexes_margin = list(itertools.chain.from_iterable(indexes_margin))

    return indexes_margin


def plot_individuals_samples(sample_by_line,
                             sample_names,
                             data,
                             label,
                             label_values,
                             folder_results_msvm):
    data.index = range(1, len(data) + 1)
    label.index = range(1, len(data) + 1)
    colors = label.replace(to_replace = label_values,
                           value = ['green','dodgerblue', 'red'])
    for sample in sample_names:
        plt.scatter(data[1].loc[sample_by_line[sample]],
                    data[0].loc[sample_by_line[sample]],
                    marker = 'o',
                    edgecolors = 'black',
                    c = colors.loc[sample_by_line[sample]])
    text_sample = label
    text_sample.loc[sample_by_line['training_1']] = 'tr_1'
    text_sample.loc[sample_by_line['training_2']] = 'tr_2'
    text_sample.loc[sample_by_line['validation']] = 'val'
    
    for individual in range(1, len(data) + 1):
        plt.text(data[1].loc[individual] + 0.001,
                 data[0].loc[individual]+ 0.0003,
                 text_sample.loc[individual])
    
    epsilon = 0.0005
    bounds_x = [-0.75, 1.1]
    bounds_y = [-1.001, -0.977]
    plt.axis([bounds_y[0]-epsilon, bounds_y[1]+epsilon, bounds_x[0]-epsilon, bounds_x[1]+epsilon])
    name_fig = folder_results_msvm + 'plot_sampling_individuals.pdf'
    plt.savefig(name_fig)
    plt.show()
    
    return 0