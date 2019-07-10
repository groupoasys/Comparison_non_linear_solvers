import logging

import Inverse_Optimization_Related_Problem.optimization_problem_utils.my_project.helpers as hp 
import pyomo.environ as pe
from pyomo.opt import TerminationCondition
from pyomo.opt.base import SolverFactory
import pdb
import pandas as pd
import time
import numpy as np
import os


 ####################################################
# Model 1
####################################################
def model1(data,
           conf,
           initial_variables_multistart):
    """ Create model as a Pyomo object

    Parameters:
        data (dict): Dictionary with pandas dataframes with all data
        conf (dict): Dictionary with model options

    Returns:
        Pyomo model instance

    """
    m = pe.ConcreteModel('MyModel')

    # Encapsulate data in model instance
    m.u = data['u']
    m.theta = data['parameters'].loc['theta', 'Value']

    # Sets
    m.i = pe.Set(initialize=m.u.index.get_level_values('i').unique(),
                 ordered=True, doc=' ')
    m.j = pe.Set(initialize=m.u.index.get_level_values('i').unique(),
                 ordered=True, doc=' ')

    # Variables
    m.x = pe.Var(m.i,
                 within=pe.Reals,
                 doc='Decision variable',
                 initialize = initial_variables_multistart['x'].to_dict())
    m.y = pe.Var(m.j,
                 within = pe.Binary,
                 doc='Decision variable',
                 initialize = initial_variables_multistart['y'].to_dict())
    
    # Constraints
    m.eq1_model1          = pe.Constraint(m.i, rule=eq1_model1_rule,                doc=' ')
    m.eq2_model1          = pe.Constraint(     rule=eq2_model1_rule,                doc=' ')
    m.eq3_model1          = pe.Constraint(     rule=eq3_model1_rule,                doc=' ')   
    m.eq4_model1          = pe.Constraint(     rule=eq4_model1_rule,                doc=' ')
    
    # Objective function
    m.obj_model1 = pe.Objective(rule=obj_model1_rule, sense=pe.minimize, doc='Objective function of model 1')
    
    
    logging.info("Model prepared")
    return m
    
########################################################
# Equations for model 1
########################################################

# Eq1
def eq1_model1_rule(m, i):
    return (m.x[i]**2)*m.x[1]<=1

# Eq2
def eq2_model1_rule(m):
    return sum(m.x[i]**2 for i in m.i)<=10

#Eq3
def eq3_model1_rule(m):
    return sum(m.x[i]**3 for i in m.i)<=sum(pe.sin(m.x[i])*pe.cos(m.x[i]) for i in m.i)

#Eq4
def eq4_model1_rule(m):
    return sum(sum(m.x[i]*m.y[j] for i in m.i) for j in m.j)<=1
    


    
# Objective Function
def obj_model1_rule(m):
    return sum( (m.theta + m.u.at[i, 'u']) * (pe.sin(m.x[i])*pe.cos(m.x[i])*m.x[i]**2)
                 for i in m.i) + sum(pe.cos(m.y[j]**2)*pe.sin(m.y[j]) for j in m.j)
    
########################################################
# Processing
########################################################

def run_solver(instance,
               conf,
               solver,
               neos_flag):
    """Method to solve a pyomo instance

    Parameters:
        instance: Pyomo unsolved instance
        solver(str): solver to use. Select between (glpk, cplex, gurobi, cbc et.c)
        solver_manager (str): serial or pyro
        tee(bool): if True a detailed solver output will be printed on screen
        options_string: options to pass to solver

    Returns:

    """
    # initialize the solver / solver manager.
#    if solver == 'minos':
#        pdb.set_trace()
    solver_name = solver
    if neos_flag:
        solver = pe.SolverManagerFactory("neos")
    else:
        solver = pe.SolverFactory(solver_name)
    if solver is None:
        raise Exception("Solver %s is not available on this machine." % solver)
        
    if neos_flag:
        results = solver.solve(instance,
                               tee = conf['tee'],
                               symbolic_solver_labels=conf['symbolic_solver_labels'], 
                               load_solutions=False,
                               opt = solver_name)
    else:
        results = solver.solve(instance,
                               tee = conf['tee'],
                               symbolic_solver_labels=conf['symbolic_solver_labels'], 
                               load_solutions=False)
    if results.solver.termination_condition not in (TerminationCondition.optimal, TerminationCondition.maxTimeLimit):
        # something went wrong
        logging.warn("Solver: %s" % results.solver.termination_condition)
        logging.debug(results.solver)
    else:
        logging.info("Model solved. Solver: %s, Time: %.2f, Gap: %s" %
                     (results.solver.termination_condition, results.solver.time, results.solution(0).gap))
        instance.solutions.load_from(results)
        instance.obj_model1.expr()
    return instance, results.solver, results.solution

def run_mymodel(config,
                neos_flag,
                iteration_multistart,
                results_dir='./results/',
                solver='cplex'):
    """ Run my model
    
    Parameters:
        config (str): configuration file
        results_dir (str): directory to store results
        solver (str): solver to use (CPLEX, glpk, etc.)

    """
    results_dir = os.path.join(results_dir,config['output_files']['folder'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    out_keys = ['info', 'theta', 'u', 'x', 'y']

    dict_out = {}
    for i in range(len(out_keys)):
        dict_out[out_keys[i]] = pd.DataFrame([]) 

    if config['model_cfg']['how_to_read_data']=='endogenously':
        logging.info("## Creating input data endogenously ##")
        data = {}
        data['parameters'] = pd.DataFrame([], index=['theta', 'n'], columns=['Value'])
        data['parameters'].loc['theta', 'Value'] = config['model_cfg']['theta'] 
        data['parameters'].loc['n', 'Value'] = config['model_cfg']['n_sample']
        data['parameters'].index.name = 'p'
    
        logging.info("## Generating input data for the parameter u ##")
        np.random.seed(seed = 1133)
        data['u'] = pd.DataFrame(np.random.uniform(-2,1,data['parameters'].loc['n', 'Value']), index=range(1, int(data['parameters'].loc['n', 'Value'])+1), columns=['u'])
        data['u'].index.name = 'i'
        
    elif config['model_cfg']['how_to_read_data']=='exogenously':
        # Reading the data
        #logging.info("## Reading input data exogenously ##")
        data = hp.parse_excel(config['input_files']['data'], config) 
             
    #logging.info("## Solving model 1 to generate x ##")
    
    
    #Initialize variables multistart
    
    initial_variables_multistart = get_initial_variables_multistart(iteration_multistart = iteration_multistart,
                                                                    config = config)
    
    # Create instance model
    instance_mymodel = model1(data = data,
                                 conf = config,
                                 initial_variables_multistart = initial_variables_multistart)
    
    start_time = time.time()
    solved_instance, solver_status, solver_solutions  = run_solver(instance_mymodel,
                                                                      config['solver_cfg'],
                                                                      solver = solver,
                                                                      neos_flag = neos_flag)
    end_time = time.time()   
    if solver_status.termination_condition.index == 8:  # Termination condition of solver: Optimal
        dict_out['theta'].loc['theta', 'Value'] = data['parameters'].loc['theta', 'Value']
        dict_out['x'].loc[:, 'x'] = hp.pyomo_to_pandas(solved_instance, 'x').iloc[:,0]
        dict_out['y'].loc[:, 'y'] = hp.pyomo_to_pandas(solved_instance, 'y').iloc[:,0]
        dict_out['u'].loc[:, 'u'] = data['u'].iloc[:, 0]
        dict_out['info'].loc['Time', 'Value'] = (end_time - start_time)
    else:
        dict_out['theta'].loc['theta', 'Value'] = data['parameters'].loc['theta', 'Value']
        dict_out['x'].loc[:, 'x'] = hp.pyomo_to_pandas(solved_instance, 'x').iloc[:,0]
        dict_out['y'].loc[:, 'y'] = hp.pyomo_to_pandas(solved_instance, 'y').iloc[:,0]
        dict_out['u'].loc[:, 'u'] = data['u'].iloc[:, 0]
        dict_out['info'].loc['Time', 'Value'] = (end_time - start_time)
    
    
    output = {}
    output['elapsed_time'] = dict_out['info'].loc['Time', 'Value']
    output['objective_value'] = solved_instance.obj_model1()
    output['dictionary_output'] = dict_out
    
    # Uncomment when the output should be saved in an output file
    #mp.dict_pandas_to_excel(dict_out, dir=results_dir, filename=config['output_files']['filename'])
    
    return output
def get_initial_variables_multistart(iteration_multistart,
                                     config):
    number_of_variables = config['model_cfg']['n_sample']
    np.random.seed(seed = 1559 + iteration_multistart)
    initial_variables_x = np.random.uniform(low = -10,
                                            high = 10,
                                            size = number_of_variables)
    initial_variables_y = np.random.randint(low = 0,
                                            high = 1 + 1,
                                            size = number_of_variables)
    initial_variables = pd.DataFrame({'x':initial_variables_x,
                                      'y': initial_variables_y},
                                     index = range(1, number_of_variables + 1))
    
    return initial_variables
