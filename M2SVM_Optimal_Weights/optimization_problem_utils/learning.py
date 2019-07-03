from sklearn import neighbors, svm, tree
import numpy as np
import time
import pandas as pd
import pdb
import sampling as smp
import math
import parameter_tuning_grid as ptg

def learn_line_separate(x_train,
                        y_train,
                        x_test,
                        y_test,
                        method,
                        level,
                        remove,
                        weights = math.nan):
    mat = []
    score = []
    mat_score = [[0,0,0],[0,0,0],[0,0,0]]
    num_total = 0
    num_trust = 0
    num_right = 0
    if method == 'multiclass_SVM':
        indexes_individuals_sample_test = pd.DataFrame(data = np.array(range(len(x_train) + 1, len(x_train) + len(x_test) + 1)),
                                                       index = range(len(x_train) + 1, len(x_train) + len(x_test) + 1),
                                                       columns = ['time_period'])
        indexes_individuals_sample_training = pd.DataFrame(data = np.array(range(1, len(x_train) + 1)),
                                                           index = range(1, len(x_train) + 1),
                                                           columns = ['time_period'])
        seed_sampling = 1455
        seed_shuffle = 1615
        sample_names = ['training_1',
                        'training_2',
                        'validation',
                        'test']
        number_of_samples_except_testing = 3
        output_sampling = smp.sampling_method(indexes_individuals_sample_training = indexes_individuals_sample_training,
                                              indexes_individuals_sample_test = indexes_individuals_sample_test,
                                              number_of_lines = y_train.shape[1],
                                              seed_sampling = seed_sampling,
                                              seed_shuffle = seed_shuffle,
                                              sample_names = sample_names,
                                              number_of_samples_except_testing = number_of_samples_except_testing)
        samples = output_sampling['indexes_individuals_all_samples_and_lines']
        correspondence_time_period_line_training_samples = output_sampling['correspondence_time_period_line_training_samples']
    for line in range(y_train.shape[1]):
      y_train_line = y_train.iloc[:,line]
      y_test_line = y_test.iloc[:,line]
      if method[0:3]=='knn':
        neig = int(method[3:5])  
        if weights.count(weights[0]) == len(weights):
          clf = neighbors.KNeighborsClassifier(neig,weights='distance')
        else:
          clf = neighbors.KNeighborsClassifier(neig,weights='distance',metric='wminkowski',p=2,metric_params={'w': np.array(weights[line])})
        clf.fit(x_train,y_train_line)
        score.append(clf.score(x_test, y_test_line, sample_weight=None))
        pred = clf.predict(x_test).tolist()
        prob = np.max(clf.predict_proba(x_test),axis=1).tolist()
      elif method=='svm':
        try:
          start = time.time()
          clf = svm.SVC(gamma='auto',probability=True)
          clf.fit(x_train,y_train_line)
          print(time.time()-start)
          score.append(clf.score(x_test, y_test_line, sample_weight=None))
          pred = clf.predict(x_test).tolist()          
          prob = np.max(clf.predict_proba(x_test),axis=1).tolist()
        except ValueError:
          pred = [y_train_line[0] for i in range(24)]
          prob = [1 for i in range(24)]
          score.append(sum(y_test==y_train_line[0])/len(y_test))
      elif method=='tree':
        clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
        clf = clf.fit(x_train,y_train_line)
        score.append(clf.score(x_test, y_test_line, sample_weight=None))
        pred = clf.predict(x_test).tolist()
        prob = np.max(clf.predict_proba(x_test),axis=1).tolist()  
      elif method == 'multiclass_SVM':
          
        print("The congestion of line %d is learnt with the multiclass SVM method" % (line + 1))
        data = pd.concat([x_train, x_test])
        data.index = range(1, len(data) + 1)
        label = pd.concat([y_train, y_test])
        label.index = range(1, len(label) + 1)
        SVM_regularization_parameter_grid = [10**range_element for range_element in range(-3, -2)]
        seed_random_prediction_values = 1451
        lowest_label_value = -1
        highest_label_value = 1
        sample_to_get_best_parameters = sample_names[2]
        maximum_number_iterations_alternating_approach = 1
        threshold_difference_objective_values_second_step = 1e-5
        default_difference_objective_values_second_step = 1e5
        seed_initialize_parameters = 1133
        number_of_nodes = 73
        bounds_weights = {'lower_bound': 0,
                          'upper_bound_initial_solution': 1e2}
        label_values = [-1, 0, 1]
        new_label_values = [1, 2, 3]
        number_of_renewable_energy = 1
        maximum_number_iterations_multistart = 3
        perturbation_multistart_variables = {'weights': 1}
        seed_multistart = 1219
        default_new_objective_value_second_step = 1e3
        best_results_tune_parameters_grid = ptg.tune_parameters_grid(SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,
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
                                                                     correspondence_time_period_line_training_samples = correspondence_time_period_line_training_samples,
                                                                     maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                                     perturbation_multistart_variables = perturbation_multistart_variables,
                                                                     seed_multistart = seed_multistart,
                                                                     default_new_objective_value_second_step = default_new_objective_value_second_step)
        
        pred = (best_results_tune_parameters_grid['prediction_all_samples']['test'][0]).tolist()
        score.append(0.5)
        prob = (0.5*np.ones(shape = len(y_test_line),dtype = float)).tolist()
      if remove == 1:
          values = [1,-1]
      elif remove == 2:
          values = [0]
      elif remove == 3:
          values = [-1,1,0]
      row = []
      for j in range(len(pred)):    
          if prob[j] >= level:
            mat_score[int(y_test_line.iloc[j])+1][int(pred[j])+1] += 1
          if pred[j] in values:
            num_total += 1
            if prob[j] >= level:
              num_trust += 1
              row.append(pred[j])
              if pred[j] == y_test_line.iloc[j]:
                num_right += 1
            else:
              row.append(2)
          else:
            row.append(2)
      mat.append(row)
    return {'mat':pd.DataFrame(mat).T,
            'score':100*sum(score)/len(score),
            'mat_score':mat_score,
            'trust':100*num_trust/num_total,
            'right':100*num_right/num_trust}






  



