import sys
import numpy as np
import pandas as pd
import M2SVM_Optimal_Weights.optimization_problem_utils.sampling as smp
import M2SVM_Optimal_Weights.optimization_problem_utils.parameter_tuning_grid as ptg
import pickle
import M2SVM_Optimal_Weights.optimization_problem_utils.normalization as nm
import os
from statistics import mean 

class sys:

  # Inizialization
  def __init__(self,gen_file,lin_file,data_file,c_shed,slack_bus=1):    
    # gen_file: file with generation data  
    # lin_file: file with transmission data
    # c_shed: cost of load shedding
    self.gen = pd.read_csv(gen_file)
    self.n_gen = len(self.gen['g'])    
    self.lin = pd.read_csv(lin_file)
    self.n_lin = len(self.lin['l'])    
    self.n_bus = max(max(self.lin['from']),max(self.lin['to']))
    self.data_file = data_file
    self.data = pd.read_csv(data_file, index_col=False)
    self.c_shed = c_shed
    self.slack_bus = slack_bus
    matrixA = [[0 for i in range(self.n_lin)] for j in range(self.n_bus)]
    matrixX = [[0 for i in range(self.n_lin)] for j in range(self.n_lin)]
    for i,r in self.lin.iterrows():
        matrixA[int(r['from'])-1][int(r['l'])-1] = 1
        matrixA[int(r['to'])-1][int(r['l'])-1] = -1
        matrixX[int(r['l'])-1][int(r['l'])-1] = r['x']
    matrixA = np.delete(np.array(matrixA),slack_bus-1, 0)
    matrixX = np.array(matrixX)
    ptdf = np.linalg.multi_dot([matrixX,matrixA.T,np.linalg.inv(np.linalg.multi_dot([matrixA,matrixX,matrixA.T]))])
    ptdf = np.insert(ptdf, slack_bus - 1, np.zeros((self.n_lin)), axis = 1)
    for b in range(self.n_bus):
      ptdf[:,b] = ptdf[:,b] - ptdf[:,slack_bus-1]
    self.ptdf = ptdf.tolist()
    self.saturation = np.round(100*np.mean(abs(self.data.ix[:,'l1':'l'+ str(self.n_lin)].values),axis=0),1).tolist()


  # function to select the training and test data and weights of features
  def learning_test_data(self,
                         ini_train,
                         end_train,
                         ini_test,
                         end_test,
                         net_demand=False,
                         weight_ptdf=False):
    self.ini_train = ini_train
    self.end_train = end_train
    self.ini_test = ini_test
    self.end_test = end_test
    self.net_demand = net_demand
    self.weight_ptdf = weight_ptdf

    if net_demand:
      x_train = pd.DataFrame(self.data.iloc[24*ini_train:24*end_train,:self.n_bus].values-self.data.iloc[24*ini_train:24*end_train,self.n_bus:2*self.n_bus].values)  
      x_test = pd.DataFrame(self.data.iloc[24*ini_test:24*end_test,:self.n_bus].values-self.data.iloc[24*ini_test:24*end_test,self.n_bus:2*self.n_bus].values)  
      if weight_ptdf:
          weights = self.ptdf #ToDo: discuss this point with Salva
          
      else:
          weights = [1 for i in range(self.n_bus)]
    else:  
      x_train = self.data.iloc[24*ini_train:24*end_train,:2*self.n_bus]
      x_test = self.data.iloc[24*ini_test:24*end_test,:2*self.n_bus].reset_index().iloc[:,1:]
      if weight_ptdf:
          weights = [self.ptdf[i] + self.ptdf[i] for i in range(self.n_lin)]
      else:
          weights = [1 for i in range(2*self.n_bus)] 
    y_train = self.data.iloc[24*ini_train:24*end_train,2*self.n_bus:]    
    y_test = self.data.iloc[24*ini_test:24*end_test,2*self.n_bus:]
    
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.weights = weights
    
    return self

  def learn_line(self,
                 method,
                 level=1,
                 net_demand = True,
                 weight_ptdf = True,
                 weights_values = np.abs([[1, 1, 1],[1, 1, 1],[1, 1, 1]]),
                 SVM_regularization_parameter_grid = [10**range_element for range_element in range(0, 1)],
                 solver = 'ipopt',
                 problem = '',
                 neos_flag = False,
                 number_of_variables = -1,
                 number_of_constraints = -1,
                 sense_opt_problem = 'min',
                 maximum_number_iterations_multistart = 1,
                 folder_results = 'folder_results_by_default'):
    self.method = method  
    if method == 'illustrative_m2svm_optimization':
        if net_demand:
            x_train = pd.DataFrame(self.data.iloc[self.ini_train:self.end_train,:self.n_bus].values-self.data.iloc[self.ini_train:self.end_train,self.n_bus:2*self.n_bus].values)  
            x_test = pd.DataFrame(self.data.iloc[self.ini_test:self.end_test,:self.n_bus].values-self.data.iloc[self.ini_test:self.end_test,self.n_bus:2*self.n_bus].values)  
            
            y_train = self.data.iloc[self.ini_train:self.end_train,2*self.n_bus:]    
            y_test = self.data.iloc[self.ini_test:self.end_test,2*self.n_bus:]
            if weight_ptdf:
                weights = np.abs(self.ptdf) 
            else:
                weights = [[1 for i in range(self.n_bus)] for line in range(self.n_lin)]
        else:  
            x_train = self.data.iloc[self.ini_train:self.end_train,:2*self.n_bus]
            x_test = self.data.iloc[self.ini_test:self.end_test,:2*self.n_bus].reset_index().iloc[:,1:]
            
            y_train = self.data.iloc[self.ini_train:self.end_train,self.n_bus:]    
            y_test = self.data.iloc[self.ini_test:self.end_test,self.n_bus:]
            if weight_ptdf:
                weights = [self.ptdf[i] + self.ptdf[i] for i in range(self.n_lin)]
            else:
                weights = [[1 for i in range(2*self.n_bus)] for line in range(self.n_lin)]
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.weights = weights
        indexes_individuals_sample_test = pd.DataFrame(data = np.array(range(len(self.x_train) + 1, len(self.x_train) + len(self.x_test) + 1)),
                                                       index = range(len(self.x_train) + 1, len(self.x_train) + len(self.x_test) + 1),
                                                       columns = ['time_period'])
        indexes_individuals_sample_training = pd.DataFrame(data = np.array(range(1, len(self.x_train) + 1)),
                                                           index = range(1, len(self.x_train) + 1),
                                                           columns = ['time_period'])
        best_results_tune_parameters_grid = {}
        seed_sampling = 1309
        seed_shuffle = 1615
        sample_names = ['training_1',
                      'training_2',
                      'validation',
                      'test']
        number_of_samples_except_testing = 3
        number_of_lines = self.y_train.shape[1]
        output_sampling = smp.sampling_method(indexes_individuals_sample_training = indexes_individuals_sample_training,
                                          indexes_individuals_sample_test = indexes_individuals_sample_test,
                                          number_of_lines = number_of_lines,
                                          seed_sampling = seed_sampling,
                                          seed_shuffle = seed_shuffle,
                                          sample_names = sample_names,
                                          number_of_samples_except_testing = number_of_samples_except_testing)
        samples = output_sampling['indexes_individuals_all_samples_and_lines']
        correspondence_time_period_line_all_samples = output_sampling['correspondence_time_period_line_all_samples']
        
        individuals_training_1 = list(range(1, 13))
        individuals_training_2 = [13, 16, 17, 20, 21, 23, 26, 27, 28, 31]
        individuals_validation = [14, 15, 18, 19, 22, 24, 25, 29, 30]
        for line in range(number_of_lines):
            samples[line]['training_1'] = individuals_training_1
            samples[line]['training_2'] = individuals_training_2
            samples[line]['validation'] = individuals_validation
        
        line_to_learn = 2
        correspondence_time_period_line_all_samples['training_1'] = pd.DataFrame(data = {'time_period': individuals_training_1,
                                                                                          'line': [line_to_learn]*len(individuals_training_1)},
                                                                                 index = range(1, len(individuals_training_1) + 1))
        correspondence_time_period_line_all_samples['training_2'] = pd.DataFrame(data = {'time_period': individuals_training_2,
                                                                                          'line': [line_to_learn]*len(individuals_training_2)},
                                                                                 index = range(1, len(individuals_training_2) + 1))
        correspondence_time_period_line_all_samples['validation'] = pd.DataFrame(data = {'time_period': individuals_validation,
                                                                                          'line': [line_to_learn]*len(individuals_validation)},
                                                                                 index = range(1, len(individuals_validation) + 1))
        
        
        data_train_normalized = nm.normalize_data(data = self.x_train)
        data_test_normalized = nm.normalize_data(data = self.x_test)
        
        
        
        seed_random_prediction_values = 1451
        lowest_label_value = -1
        highest_label_value = 1
        sample_to_get_best_parameters = sample_names[2]
        maximum_number_iterations_alternating_approach = 0
        threshold_difference_objective_values_second_step = 1e-5
        default_difference_objective_values_second_step = 1e5
        seed_initialize_parameters = 1133
        number_of_nodes = self.n_bus
        bounds_weights = {'lower_bound': 0,
                         'upper_bound_initial_solution': 10}
        label_values = [-1, 0, 1]
        new_label_values = [1, 2, 3]
        if net_demand:
            number_of_renewable_energy = 0
        else:
            number_of_renewable_energy = 1
        perturbation_multistart_variables = {'weights': 1}
        seed_multistart = 1219
        default_new_objective_value_second_step = 1e3
        beggining_file_name_to_save_results = 'results_by_line_'
        folder_results_msvm = folder_results
        if not (os.path.isdir('./' + folder_results_msvm)):
            os.mkdir(folder_results_msvm)
      
        if weight_ptdf:
            approach = 'ptdf'
        else:
            approach = 'random'
        
        
        ######################################################################################
        # This code is not necessary in the solvers comparison
#        file_to_write = open(csv_file, 'a')
#        file_to_write.write('data_file' + ',' + 'line' + ',' +"%zeros" +',' +"%ones" +',' +"%minus_ones" +',' + 'approach' + ',' + '% total accuracy' + ',' + 'SVM reg param' + ',' +'weights'+'\n')
#        file_to_write.close()
#        
#        pltu.plot_individuals_samples(sample_by_line = samples[line],
#                                      sample_names = ['training_1', 'training_2', 'validation'],
#                                      data = data_train_normalized,
#                                      label = y_train.copy().iloc[:,line_to_learn - 1],
#                                      label_values = label_values,
#                                      folder_results_msvm = folder_results_msvm)
        ######################################################################################
        for line in range(1, 2):
            data = pd.concat([data_train_normalized, data_test_normalized])
            data.index = range(1, len(data) + 1)
            label = pd.concat([self.y_train, self.y_test])
            label.index = range(1, len(label) + 1)
            
            if weight_ptdf:
               initial_weights = np.absolute(np.array(self.ptdf[line]))
            else:
               initial_weights = [None]
            best_results_tune_parameters_grid[line] = ptg.tune_parameters_grid(SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,
                                                                           sample_by_line = samples[line],
                                                                           label = label,
                                                                           line = line,
                                                                           sample_names = sample_names,
                                                                           seed_random_prediction_values = seed_random_prediction_values,
                                                                           lowest_label_value = lowest_label_value,
                                                                           highest_label_value = highest_label_value,
                                                                           sample_to_get_best_parameters = sample_to_get_best_parameters,
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
                                                                           initial_weights = initial_weights,
                                                                           solver = solver)
            
            file_name_to_save_results = folder_results_msvm + beggining_file_name_to_save_results + str(line + 1) +'_' + approach+'.pydata'          
            file_to_save = open(file_name_to_save_results, 'wb')
            pickle.dump(best_results_tune_parameters_grid[line], file_to_save)
            file_to_save.close()
        
    return best_results_tune_parameters_grid
        
        
    
    
    
    
    
    
    
  
