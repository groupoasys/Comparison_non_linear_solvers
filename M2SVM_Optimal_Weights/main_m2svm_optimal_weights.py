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


problem = 'm2svm_optimal_weights'
number_of_variables = 15
number_of_constraints = 36
sense_opt_problem = 'min'    
maximum_number_iterations_multistart = 1000
folder_results = 'results_' + problem + '/'
csv_file_name_multistart = 'results_multistart'
csv_file_summary_results = 'summary_results'

solvers_list_ampl = ['conopt',
                    'loqo',
                    'minos',
                    'snopt']
solvers_list_neos_flag_false = ['ipopt',
                                'bonmin',
                                'couenne']
solvers_list_neos_flag_true = ['conopt',
                               'ipopt',
                               'filter',
                               'knitro',
                               'loqo',
                               'minos',
                               'mosek',
                               'snopt',
                               'bonmin',
                               'couenne',
                               'filmint']

cu.create_folder_results_if_it_doesnt_exits(folder_results = folder_results)
cu.initialize_summary_results_file(folder_results = folder_results,
                                   csv_file_summary_results = csv_file_summary_results)
for solver in solvers_list_ampl:
    neos_flag = False
    ampl_flag = True
    cu.run_optimization_problem_given_solver(solver = solver,
                                             problem = problem,
                                             neos_flag = neos_flag,
                                             number_of_variables = number_of_variables,
                                             number_of_constraints = number_of_constraints,
                                             sense_opt_problem = sense_opt_problem,
                                             maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                             folder_results = folder_results,
                                             csv_file_name_multistart = csv_file_name_multistart,
                                             ampl_flag = ampl_flag)

for solver in solvers_list_neos_flag_false:
    neos_flag = False
    ampl_flag = False
    cu.run_optimization_problem_given_solver(solver = solver,
                                             problem = problem,
                                             neos_flag = neos_flag,
                                             number_of_variables = number_of_variables,
                                             number_of_constraints = number_of_constraints,
                                             sense_opt_problem = sense_opt_problem,
                                             maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                             folder_results = folder_results,
                                             csv_file_name_multistart = csv_file_name_multistart,
                                             ampl_flag = ampl_flag)

for solver in solvers_list_neos_flag_true:
    neos_flag = True
    ampl_flag = False
    print(solver)
    cu.run_optimization_problem_given_solver(solver = solver,
                                             problem = problem,
                                             neos_flag = neos_flag,
                                             number_of_variables = number_of_variables,
                                             number_of_constraints = number_of_constraints,
                                             sense_opt_problem = sense_opt_problem,
                                             maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                             folder_results = folder_results,
                                             csv_file_name_multistart = csv_file_name_multistart,
                                             ampl_flag = ampl_flag)
