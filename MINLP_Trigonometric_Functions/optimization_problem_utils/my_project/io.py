import logging
import os
from datetime import datetime
import yaml
import pandas as pd

## Preprocessing

def read_yaml(filename):
    """ Loads YAML file to dictionary"""
    with open(filename, 'r') as f:
        try:
            return yaml.load(f)
        except yaml.YAMLError as exc:
            logging.error('Cannot parse config file: {}'.format(filename))
            raise

def parse_excel(filename, config):
    """Read Excel and prepare input pandas

    Parameters:
        filename: filename of excel file according to template

    Returns:
        Dictionary with input ready to be processed by Pyomo
    """
    
    info = {'filename': os.path.splitext(os.path.split(filename)[1])[0],
            'created_time': datetime.now().strftime('%Y%m%dT%H%M')
            }
                  
    with pd.ExcelFile(filename) as f:
        u = f.parse('u').set_index(['i'])
        parameters = f.parse('parameters').set_index(['p'])
            
        data = {'u': u,
                'parameters': parameters,
                'info': info
                }
            
    logging.info("Data imported")

    return data
    

def save(instance, filename):
    """Save model instance to pickle file.

    Parameters:
        instance: a model instance
        filename: pickle file to be written
    """
    import pickle
    with open(filename, 'wb') as pf:  # use gzip if it gets too big
        pickle.dump(instance, pf, pickle.HIGHEST_PROTOCOL)
        #cPickle.dump(instance, pf, cPickle.HIGHEST_PROTOCOL)
    logging.info("Model saved")


def load(filename):
    """Load a model instance from a pickle file

    Parameters:
        filename: pickle file

    Returns:
        the unpickled model instance

    """
    
    import pickle
    with open(filename, 'rb') as pf:
        return pickle.load(pf)

