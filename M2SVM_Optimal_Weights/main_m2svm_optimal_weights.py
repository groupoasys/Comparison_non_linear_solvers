# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:53:25 2019

@author: Asun
"""

from __future__ import division

for name in dir():
    if not name.startswith('_'):
        del globals()[name]


import os

directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(directory_path, os.path.pardir))

import comparison_utils as cu

os.chdir(directory_path)

solver = 'conopt'
problem = 'm2svm_optimal_weights'
neos_flag = True
number_of_variables = -1
number_of_constraints = -1
sense_opt_problem = 'min'    
maximum_number_iterations_multistart = 2
folder_results = 'results_' + problem + '/'
csv_file_name_multistart = 'results_multistart'
csv_file_summary_results = 'summary_results'

cu.run_optimization_problem_given_solver(solver = solver,
                                         problem = problem,
                                         neos_flag = neos_flag,
                                         number_of_variables = number_of_variables,
                                         number_of_constraints = number_of_constraints,
                                         sense_opt_problem = sense_opt_problem,
                                         maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                         folder_results = folder_results,
                                         csv_file_name_multistart = csv_file_name_multistart)
