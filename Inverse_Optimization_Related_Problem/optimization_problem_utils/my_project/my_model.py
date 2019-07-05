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
    m.j = pe.Set(initialize=m.u.index.get_level_values('i').unique(),
                 ordered=True, doc=' ')
           
    # Variables
    m.x = pe.Var(m.i,
                 within=pe.Reals,
                 doc='Decision variable',
                 initialize = 1)
    m.y = pe.Var(m.j,
                 within = pe.Binary,
                 doc='Decision variable',
                 initialize = 1)
    
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
               solver):
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
    solver = SolverFactory(solver)
    if solver is None:
        raise Exception("Solver %s is not available on this machine." % solver)
    
    results = solver.solve(instance,  #TODO: try with **conf
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
        instance.obj_model1.expr()
    return instance, results.solver, results.solution
