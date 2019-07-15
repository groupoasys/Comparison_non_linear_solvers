import logging
import os

import pandas as pd
import pdb

def to_bool(boolean):
    return 1 if boolean == 'YES' else 0

def pandas_to_excel(dataframe, dir='./results/', filename='pandas.xlsx'):
    """Convert a pandas dataframe to excel file

    Parameters:
        dataframe (df): pandas dataframe to be exported
        dir (str): directory to store the excel file
        filename (str): filename of excelfile
    """
    resultfile = os.path.join(dir, filename)
    dataframe.to_excel(resultfile)
    
def pandas_to_csv(dataframe, dir='./results/', filename='pandas.csv'):
    """Convert a pandas dataframe to csv file

    Parameters:
        dataframe (df): pandas dataframe to be exported
        dir (str): directory to store the csv file
        filename (str): filename of csvfile
    """
    resultfile = os.path.join(dir, filename)
    dataframe.to_csv(resultfile, sep=';')

def dict_to_excel(dictionary, dir='./results/', filename='dict.xlsx'):
    """Convert a dictionary of sclaras to excel file

    Parameters:
        dictionary (dict): dictionary to be exported
        dir (str): directory to store the excel file
        filename (str): filename of excelfile
    """
    resultfile = os.path.join(dir, filename)
    a = pd.DataFrame.from_dict(dictionary, orient='index')
    a.to_excel(resultfile)


def dict_pandas_to_excel(dictionary, dir='./results/', filename='dict.xlsx'):
    """Convert a dictionary of pandas to excel file

    Parameters:
        dictionary (dict): dictionary to be exported
        dir (str): directory to store the excel file
        filename (str): filename of excelfile
    """
    resultfile = os.path.join(dir, filename)

    writer = pd.ExcelWriter(resultfile)
    for i, v in dictionary.items():
        v.to_excel(writer, '%s' % i, merge_cells=False)
    writer.save()  # save

## HELPERS: PYOMO


def get_sets(instance, var):
    """Get sets that belong to a pyomo Variable or Param

    Parameters:
        instance: Pyomo Instance
        var: Pyomo Var (or Param)

    Returns:
        A list with the sets that belong to this Param
    """
    sets = []
    var = getattr(instance, var)

    if var.dim() > 1:
        for pset in var._index.set_tuple:
            sets.append(pset.name)
    else:
        sets.append(var._index.name)
    return sets


def get_set_members(instance, sets):
    """Get set members that belong to this set

    Parameters:
        instance: Pyomo Instance
        sets: Pyomo Set

    Returns:
        A list with the set members
    """
    
    sm = []
    for s in sets:
        sm.append([v for v in getattr(instance, s).value])
    return sm


def pyomo_to_pandas(instance, var):
    """Function converting a pyomo variable or parameter into a pandas dataframe.
    The variable must have one, two, or three dimensions and the sets must be
    provided as a list of lists.

    Parameters:
        instance: Pyomo Instance
        var: Pyomo variable

    Returns:
        Instance in pandas Dataframe format

    """
    
    
    setnames = get_sets(instance, var)
    sets = get_set_members(instance, setnames)
    var = getattr(instance, var)  # Previous script used model.var instead of var
    #### FIXME
    if len(sets) != var.dim():
        raise Exception('The number of provided set lists (%s) does not match the dimensions of the variable (%s)'
                        % (str(len(sets)), str(var.dim()))
                        )
    if var.dim() == 1:
        [SecondSet] = sets
        out = pd.DataFrame(columns=[var.name], index=SecondSet)
        data = var.get_values()
        for idx in data:
            out[var.name][idx] = data[idx]
        return out

    elif var.dim() == 2:
        [FirstSet, SecondSet] = sets
        out = pd.DataFrame(columns=SecondSet, index=FirstSet)
        data = var.get_values()
        for idx in data:
            out[idx[1]][idx[0]] = data[idx]
        return out
        
    elif var.dim() == 3:
        [FirstSet, SecondSet, ThirdSet] = sets
        out = pd.DataFrame(columns=SecondSet, index=pd.MultiIndex.from_product([ThirdSet,FirstSet]))
        data = var.get_values()
        for idx in data: 
            out[idx[1]][idx[2], idx[0]] = data[idx]
        return out
        
    else:
        raise Exception('the pyomo_to_pandas function currently only accepts one-, two-, or three-dimensional variables')

def pyomo_to_pandas_const(instance, const):
    """Function converting a dual variable associated with a constraint into a pandas dataframe.
    The dual variable must have one, two, or three dimensions and the sets must be
    provided as a list of lists.

    Parameters:
        instance: Pyomo Instance
        const: Pyomo constraint

    Returns:
        Instance in pandas Dataframe format

    """
    # TODO: raise exception if the problem is mixed-integer linear problem since
    # pyomo only can get dual variables from a linear problem
    
    setnames = get_sets(instance, const)
    sets = get_set_members(instance, setnames)
    const = getattr(instance, const)
    
    if const.dim() == 1:
        [SecondSet] = sets
        out = pd.DataFrame(columns=[const.name], index=SecondSet)
        for idx in const:
            out[const.name][idx] = instance.dual[const[idx]]
        return out

    elif const.dim() == 2:
        [FirstSet, SecondSet] = sets
        out = pd.DataFrame(columns=SecondSet, index=FirstSet)
        for idx in const:
            out[idx[1]][idx[0]] = instance.dual[const[idx]]
        return out
        
    elif const.dim() == 3:
        [FirstSet, SecondSet, ThirdSet] = sets
        out = pd.DataFrame(columns=SecondSet, index=pd.MultiIndex.from_product([ThirdSet, FirstSet]))
        for idx in const: 
            out[idx[1]][idx[2], idx[0]] = instance.dual[const[idx]]
        return out
        
    else:
        raise Exception('the pyomo_to_pandas function currently only accepts one-, two-, or three-dimensional variables')
    
def pyomo_param_to_pandas(instance, indexedparam)  :
    """Function converting an indexed parameter into a pandas dataframe

    Parameters:
        instance: Pyomo Instance
        indexedparam: Pyomo indexed parameter

    Returns:
        Instance in pandas Dataframe format

    """
    setnames = get_sets(instance, indexedparam)
    sets = get_set_members(instance, setnames)
    indexedparam = getattr(instance, indexedparam)
    
    if indexedparam.dim() == 1:
        [SecondSet] = sets
        out = pd.DataFrame(columns=[indexedparam.name], index=SecondSet)
        for idx in indexedparam:
            out[indexedparam.name][idx] = indexedparam[idx]
        return out
    
    if indexedparam.dim() == 2:
        [FirstSet, SecondSet] = sets
        out = pd.DataFrame(columns=SecondSet, index=FirstSet)
        for idx in indexedparam:
            out[idx[1]][idx[0]] = indexedparam[idx]
        return out

