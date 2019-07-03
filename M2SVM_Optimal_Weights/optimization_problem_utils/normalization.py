# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:24:00 2019

@author: Asun
"""
import pdb

def normalize_data(data):
    maximum_value = data.max().max()
    minimum_value = data.min().min()
    
    normalized_data = (2/(maximum_value - minimum_value))*data - ((maximum_value + minimum_value)/(maximum_value - minimum_value))
    
    return normalized_data