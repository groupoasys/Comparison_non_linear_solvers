# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:53:25 2019

@author: Asun
"""

from __future__ import division

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import comparison_utils as cu
import os

directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)

solver = 'conopt'
problem = 'm2svm_optimal_weights'
neos_flag = True
number_of_variables = -1
number_of_constraints = -1
sense_opt_problem = 'min'    

cu.run_optimization_problem_given_solver(solver = solver,
                                      problem = problem,
                                      neos_flag = neos_flag,
                                      number_of_variables = number_of_variables,
                                      number_of_constraints = number_of_constraints,
                                      sense_opt_problem = sense_opt_problem)







