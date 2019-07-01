from __future__ import division

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import os
from system import sys


directory_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory_path)

ini_train = 0 
end_train = 300
ini_test = 300
end_test = 360
n_bus = 73
directory_data = '73bus'
source_data_file = 'data360_1.csv'
generators_data_file = 'gen96.csv'
lines_data_file = 'lin96_1.csv'
data_file_path = os.path.join(directory_data, source_data_file)
generators_file_path = os.path.join(directory_data, generators_data_file)
lines_file_path = os.path.join(directory_data, lines_data_file)
method = 'multiclass_SVM'
level = -1
load_shedding = 1000
net_demand_flag = True
weight_ptdf_flag = True


sys2 = sys(gen_file = generators_file_path,
           lin_file = lines_file_path,
           data_file = data_file_path,
           c_shed = load_shedding )

sys2.learning_test_data(ini_train = ini_train,
                        end_train = end_train,
                        ini_test = ini_test,
                        end_test = end_test,
                        periods = 1,
                        net_demand = net_demand_flag,
                        weight_ptdf = weight_ptdf_flag)

sys2.learn_line(method = method,
                level = level,
                net_demand = net_demand_flag,
                weight_ptdf = weight_ptdf_flag)


