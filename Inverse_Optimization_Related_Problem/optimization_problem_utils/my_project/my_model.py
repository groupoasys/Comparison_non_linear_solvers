import logging

import pyomo.environ as pe
from pyomo.opt import TerminationCondition
from pyomo.opt.base import SolverFactory

 ####################################################
# Model 1
####################################################
def model1(data, conf):
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
           
    # Variables
    m.x = pe.Var(m.i,
                 within=pe.Reals,
                 doc='Decision variable')
    
    # Constraints
    m.eq1_model1          = pe.Constraint(m.i, rule=eq1_model1_rule,                doc=' ')
    m.eq2_model1          = pe.Constraint(m.i, rule=eq2_model1_rule,                doc=' ')      

    # Objective function
    m.obj_model1 = pe.Objective(rule=obj_model1_rule, sense=pe.minimize, doc='Objective function of model 1')

    logging.info("Model prepared")
    return m
    
########################################################
# Equations for model 1
########################################################

# Eq1
def eq1_model1_rule(m, i):
    return m.x[i] >= -1 

# Eq2
def eq2_model1_rule(m, i):
    return m.x[i] <= 1 
    
# Objective Function
def obj_model1_rule(m):
    return sum( (m.theta + m.u.at[i, 'u']) * m.x[i]  
                 for i in m.i)
    
########################################################
# Processing
########################################################

def run_solver(instance, conf):
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
    solver = SolverFactory(conf['solver'])
    if solver is None:
        raise Exception("Solver %s is not available on this machine." % solver)
    
    solver.options['timelimit'] = conf['options_timelimit']
    results = solver.solve(instance,  #TODO: try with **conf
                           options_string=conf['options_string'],
                           tee=conf['tee'],
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
        
    return instance, results.solver, results.solution
