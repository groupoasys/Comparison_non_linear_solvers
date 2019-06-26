# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:43:24 2019

@author: Asun
"""

import pdb
import numpy as np
import random
import error_handling as error
import pandas as pd

def sampling_method(indexes_individuals_sample_training,
                    indexes_individuals_sample_test,
                    number_of_lines,
                    seed_sampling,
                    seed_shuffle,
                    sample_names,
                    number_of_samples_except_testing):
    remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples = len(indexes_individuals_sample_training)%number_of_samples_except_testing
    indexes_individuals_all_lines = {}
    indexes_individuals = {}
    number_of_individuals = {}
    correspondence_time_period_line_all_samples = {}
    indexes_individuals_all_samples_and_lines = []
    for sample_name in sample_names:
        number_of_individuals = get_number_of_individual_by_sample(sample_name = sample_name,
                                                                   remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples = remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples,
                                                                   indexes_individuals_sample_training = indexes_individuals_sample_training,
                                                                   indexes_individuals_sample_test = indexes_individuals_sample_test,
                                                                   number_of_samples_except_testing = number_of_samples_except_testing,
                                                                   number_of_individuals = number_of_individuals)
        indexes_individuals = get_indexes_individuals_by_sample(sample_name = sample_name,
                                                                seed_sampling = seed_sampling,
                                                                indexes_individuals_sample_training = indexes_individuals_sample_training,
                                                                indexes_individuals_sample_test = indexes_individuals_sample_test,
                                                                number_of_individuals = number_of_individuals,
                                                                indexes_individuals = indexes_individuals)
        indexes_individuals_all_lines = get_indexes_individuals_all_lines(sample_name = sample_name,
                                                                          indexes_individuals_all_lines = indexes_individuals_all_lines,
                                                                          indexes_individuals = indexes_individuals,
                                                                          number_of_lines = number_of_lines,
                                                                          seed_shuffle = seed_shuffle)
        
    indexes_individuals_all_samples_and_lines = get_indexes_individuals_all_samples_and_lines(number_of_lines = number_of_lines,
                                                                                              sample_names = sample_names,
                                                                                              indexes_individuals_all_lines = indexes_individuals_all_lines,
                                                                                              indexes_individuals_all_samples_and_lines = indexes_individuals_all_samples_and_lines)

    correspondence_time_period_line_all_samples = get_correspondence_time_period_line_all_samples(correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                                                  number_of_lines = number_of_lines,
                                                                                                  indexes_individuals_all_lines = indexes_individuals_all_lines,
                                                                                                  indexes_individuals_all_samples_and_lines = indexes_individuals_all_samples_and_lines,
                                                                                                  sample_names = sample_names)

    output_sampling = {'indexes_individuals_all_samples_and_lines': indexes_individuals_all_samples_and_lines,
                       'correspondence_time_period_line_all_samples': correspondence_time_period_line_all_samples}
    return output_sampling

def list_split(list_to_split,
               number_of_parts,
               seed_shuffle):
    random.seed(a = seed_shuffle)
    random.shuffle(list_to_split)
    length_list = len(list_to_split)
    number_of_elements_each_part, number_of_elements_left = divmod(length_list, number_of_parts)
    list_of_parts = [list_to_split[number_of_elements_each_part*part:number_of_elements_each_part*(part+1)] for part in range(number_of_parts)]
    total_number_of_elements_divided = number_of_elements_each_part*number_of_parts
    for element in range(number_of_elements_left):
        list_of_parts[element%number_of_parts].append(list_to_split[total_number_of_elements_divided + element])  
    return list_of_parts
    
    
def get_number_of_individual_by_sample(sample_name,
                                       remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples,
                                       indexes_individuals_sample_training,
                                       indexes_individuals_sample_test,
                                       number_of_samples_except_testing,
                                       number_of_individuals):
    if remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples == 0:
            if sample_name == 'training_1' or sample_name == 'training_2' or sample_name == 'validation':
                number_of_individuals[sample_name] = int(len(indexes_individuals_sample_training)/number_of_samples_except_testing)
            elif sample_name == 'test':
                number_of_individuals[sample_name] = len(indexes_individuals_sample_test)
            else:
                raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")
    elif remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples == 1:
            if sample_name == 'training_1':
                number_of_individuals[sample_name] = int(len(indexes_individuals_sample_training)/number_of_samples_except_testing) + 1
            elif sample_name == 'training_2' or sample_name == 'validation':
                number_of_individuals[sample_name] = int(len(indexes_individuals_sample_training)/number_of_samples_except_testing)
            elif sample_name == 'test':
                number_of_individuals[sample_name] = len(indexes_individuals_sample_test)
            else:
                raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")  

    elif remainder_division_number_individuals_all_samples_except_testing_and_number_of_samples == 2:
            if sample_name == 'training_1' or sample_name == 'training_2':
                number_of_individuals[sample_name] = int(len(indexes_individuals_sample_training)/number_of_samples_except_testing) + 1
            elif sample_name == 'validation':
                number_of_individuals[sample_name] = int(len(indexes_individuals_sample_training)/number_of_samples_except_testing)
            elif sample_name == 'test':
                number_of_individuals[sample_name] = len(indexes_individuals_sample_test)
            else:
                raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")
    return number_of_individuals
    

def get_indexes_individuals_by_sample(sample_name,
                                      seed_sampling,
                                      indexes_individuals_sample_training,
                                      indexes_individuals_sample_test,
                                      number_of_individuals,
                                      indexes_individuals):
    np.random.seed(seed = seed_sampling)
    if sample_name == 'training_1':
        indexes_individuals[sample_name] = pd.DataFrame(data = sorted(np.random.choice(a = indexes_individuals_sample_training['time_period'],
                                                                                       size = number_of_individuals[sample_name],
                                                                                       replace = False)),
                                                        columns = ['time_period'],
                                                        index = np.array(range(1, number_of_individuals[sample_name] + 1)))
    elif sample_name == 'training_2':
        difference_all_individuals_and_individuals_training_1 = np.setdiff1d(ar1 = np.array(indexes_individuals_sample_training),
                                                                             ar2 = indexes_individuals['training_1'])
        indexes_individuals[sample_name] = pd.DataFrame(data = sorted(np.random.choice(a = np.array(difference_all_individuals_and_individuals_training_1),
                                                                                       size = number_of_individuals[sample_name],
                                                                                       replace = False)),
                                                        columns = ['time_period'],
                                                        index = np.array(range(1, number_of_individuals[sample_name] + 1)))
    elif sample_name == 'validation':
        difference_all_individuals_and_individuals_training_1 = np.setdiff1d(ar1 = np.array(indexes_individuals_sample_training),
                                                                             ar2 = indexes_individuals['training_1'])
        indexes_individuals[sample_name] = pd.DataFrame(data = np.setdiff1d(ar1 = difference_all_individuals_and_individuals_training_1,
                                                                            ar2 = indexes_individuals['training_2']),
                                                        columns = ['time_period'],
                                                        index = np.array(range(1, number_of_individuals[sample_name] + 1)))
    elif sample_name == 'test':
        indexes_individuals[sample_name] = indexes_individuals_sample_test
        indexes_individuals[sample_name].index = range(1, number_of_individuals[sample_name] + 1)
    else:
        raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")
        
    return indexes_individuals


def get_indexes_individuals_all_lines(sample_name,
                                      indexes_individuals_all_lines,
                                      indexes_individuals,
                                      number_of_lines,
                                      seed_shuffle):
    if sample_name == 'training_1' or sample_name == 'training_2':
        indexes_individuals_all_lines[sample_name] = list_split(list_to_split = indexes_individuals[sample_name]['time_period'].values.tolist(),
                                                                number_of_parts = number_of_lines,
                                                                seed_shuffle = seed_shuffle)
    elif sample_name == 'validation' or sample_name == 'test':
        indexes_individuals_all_lines[sample_name] = [indexes_individuals[sample_name]['time_period'].values.tolist()]*number_of_lines
    else:
        raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")
        
    return indexes_individuals_all_lines


def get_indexes_individuals_all_samples_and_lines(number_of_lines,
                                                  sample_names,
                                                  indexes_individuals_all_lines,
                                                  indexes_individuals_all_samples_and_lines):
    for line in range(0, number_of_lines):
        samples_by_line ={}
        for sample_name in sample_names:
            samples_by_line[sample_name] = sorted(indexes_individuals_all_lines[sample_name][line])
        indexes_individuals_all_samples_and_lines.append(samples_by_line)
    return indexes_individuals_all_samples_and_lines

def get_correspondence_time_period_line_all_samples(correspondence_time_period_line_all_samples,
                                                    number_of_lines ,
                                                    indexes_individuals_all_lines,
                                                    indexes_individuals_all_samples_and_lines,
                                                    sample_names):
    for sample_name in sample_names:
        correspondence_time_period_line_all_samples[sample_name] = pd.DataFrame(index = [],
                                                                                columns = ['time_period', 'line'])   
            
        if sample_name in ['training_1', 'training_2']:
            for line in range(0, number_of_lines):
                correspondence_time_period_line_by_sample = pd.DataFrame(index = range(1, len(indexes_individuals_all_lines[sample_name][line]) + 1),
                                                                         columns = ['time_period', 'line'])
                for time_period in range(1, len(indexes_individuals_all_samples_and_lines[line][sample_name]) + 1):
                    correspondence_time_period_line_by_sample.at[time_period, 'time_period'] = indexes_individuals_all_samples_and_lines[line][sample_name][time_period - 1]
                    correspondence_time_period_line_by_sample.at[time_period, 'line'] = line + 1
                correspondence_time_period_line_all_samples[sample_name] = correspondence_time_period_line_all_samples[sample_name].append(correspondence_time_period_line_by_sample).sort_values(by = 'time_period')
                correspondence_time_period_line_all_samples[sample_name].index = range(1, len(correspondence_time_period_line_all_samples[sample_name]) + 1)    
 
        elif sample_name in ['validation', 'test']:
            default_line = 0 #Since in the validation and testing set, all the individuals belong to all the lines
            default_value_for_line = -1 #In order to indicate that all the individuals share the same lines.
            correspondence_time_period_line_by_sample = pd.DataFrame(index = range(1, len(indexes_individuals_all_lines[sample_name][default_line]) + 1),
                                                                     columns = ['time_period', 'line'])
            for time_period in range(1, len(indexes_individuals_all_samples_and_lines[default_line][sample_name]) + 1):
                correspondence_time_period_line_by_sample.at[time_period, 'time_period'] = indexes_individuals_all_samples_and_lines[default_line][sample_name][time_period - 1]
                correspondence_time_period_line_by_sample.at[time_period, 'line'] = default_value_for_line
            
            correspondence_time_period_line_all_samples[sample_name] = correspondence_time_period_line_all_samples[sample_name].append(correspondence_time_period_line_by_sample).sort_values(by = 'time_period')
            correspondence_time_period_line_all_samples[sample_name].index = range(1, len(correspondence_time_period_line_all_samples[sample_name]) + 1)    
 
        else:
            raise error.my_custom_error("The name of the sample introduced does not correspond to the given configuration. Please, have a look at the sample_name variable")
                
    return correspondence_time_period_line_all_samples
    





