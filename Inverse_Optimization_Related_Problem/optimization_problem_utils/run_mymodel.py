#!/usr/bin/env python
""" Main script to run routines from my_project library
"""
from __future__ import division
import my_project as mp
import logging
import pandas as pd
import numpy as np
import os
import pdb
import time

def run_mymodel(config, results_dir='./results/', solver='cplex'):
    """ Run my model
    
    Parameters:
        config (str): configuration file
        results_dir (str): directory to store results
        solver (str): solver to use (CPLEX, glpk, etc.)

    """
    results_dir = os.path.join(results_dir,config['output_files']['folder'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    out_keys = ['info', 'theta', 'x', 'u']

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
        data['u'] = pd.DataFrame(np.random.uniform(-2,1,data['parameters'].loc['n', 'Value']), index=range(1, int(data['parameters'].loc['n', 'Value'])+1), columns=['u'])
        data['u'].index.name = 'i'
        
    elif config['model_cfg']['how_to_read_data']=='exogenously':
        # Reading the data
        logging.info("## Reading input data exogenously ##")
        data = mp.parse_excel(config['input_files']['data'], config) 
             
    logging.info("## Solving model 1 to generate x ##")
    start_time = time.time()
    
    # Create instance model
    instance_mymodel = mp.model1(data, config)
        
    solved_instance, solver_status, solver_solutions  = mp.run_solver(instance_mymodel, config['solver_cfg'])
        
    print("--- %s seconds ---" % (time.time() - start_time))
      
    if solver_status.termination_condition.index == 8:  # Termination condition of solver: Optimal
        dict_out['theta'].loc['theta', 'Value'] = data['parameters'].loc['theta', 'Value']
        dict_out['x'].loc[:, 'x'] = mp.pyomo_to_pandas(solved_instance, 'x').iloc[:,0]
        dict_out['u'].loc[:, 'u'] = data['u'].iloc[:, 0]
        dict_out['info'].loc['Time', 'Value'] = (time.time() - start_time)

    mp.dict_pandas_to_excel(dict_out, dir=results_dir, filename=config['output_files']['filename'])

def main():
    config = mp.read_yaml('./configs/config_model1.yml')
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%y %H:%M:%S',
                        filename=config['log_file'],
                        level=logging.DEBUG)

    run_mymodel(config)

if __name__ == '__main__':
    main()