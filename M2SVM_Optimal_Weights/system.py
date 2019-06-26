import pyomo.environ as pe
import sys
import random
import numpy as np
import pdb
import time
import logging
from pyomo.opt import TerminationCondition
import pandas as pd
from sklearn import neighbors, svm, tree
from sklearn.cluster import KMeans
from datetime import datetime
from learning import learn_line_separate
import sampling as smp
import parameter_tuning_grid as ptg
import matplotlib.pyplot as plt
import pickle
import normalization as nm
import os
import first_step_alternating_approach as fsap
import parameter_tuning_grid as ptg
import error_handling as error
import prediction as prediction
import plot_utils as pltu

class sys:

  # Inizialization
  def __init__(self,gen_file,lin_file,data_file,c_shed,slack_bus=1):    
    # gen_file: file with generation data  
    # lin_file: file with transmission data
    # c_shed: cost of load shedding
    self.gen = pd.read_csv(gen_file)
    self.n_gen = len(self.gen['g'])    
    self.lin = pd.read_csv(lin_file)
    self.n_lin = len(self.lin['l'])    
    self.n_bus = max(max(self.lin['from']),max(self.lin['to']))
    self.data_file = data_file
    self.data = pd.read_csv(data_file, index_col=False)
    self.c_shed = c_shed
    self.slack_bus = slack_bus
    matrixA = [[0 for i in range(self.n_lin)] for j in range(self.n_bus)]
    matrixX = [[0 for i in range(self.n_lin)] for j in range(self.n_lin)]
    for i,r in self.lin.iterrows():
        matrixA[int(r['from'])-1][int(r['l'])-1] = 1
        matrixA[int(r['to'])-1][int(r['l'])-1] = -1
        matrixX[int(r['l'])-1][int(r['l'])-1] = r['x']
    matrixA = np.delete(np.array(matrixA),slack_bus-1, 0)
    matrixX = np.array(matrixX)
    ptdf = np.linalg.multi_dot([matrixX,matrixA.T,np.linalg.inv(np.linalg.multi_dot([matrixA,matrixX,matrixA.T]))])
    ptdf = np.insert(ptdf, slack_bus - 1, np.zeros((self.n_lin)), axis = 1)
    for b in range(self.n_bus):
      ptdf[:,b] = ptdf[:,b] - ptdf[:,slack_bus-1]
    self.ptdf = ptdf.tolist()
    self.saturation = np.round(100*np.mean(abs(self.data.ix[:,'l1':'l'+ str(self.n_lin)].values),axis=0),1).tolist()

  #To create syntetic data using a covariance matrix
  def create_data_cov(self,meancov_file,n_per):    
    # meancov_file: file with averages and covariances
    # n_per: number of periods
    meancov = pd.read_csv(meancov_file)
    mean = meancov['mean']
    cov = meancov.drop(['name','mean'],axis=1).values.tolist()    
    data = pd.DataFrame(np.round(np.random.multivariate_normal(mean, cov, n_per),2))
    data.columns = ['d'+str(i+1) for i in range(self.n_bus)] + ['w'+str(i+1) for i in range(self.n_bus)]            
    self.data = data

  #To create syntetic data from historic data
  def create_data(self,load_file,wind_file,data_file,n_days=360):
    # load_file: file with demand data
    # wind_file: file with wind data
    # data_file: name of the output file with the created data
    # n_days: numbers of days we want data for
    # mix: if true, it randomly mixes demand and wind data
    random.seed(9001)
    load = pd.read_csv(load_file)
    wind = pd.read_csv(wind_file)
    mat = []
    for d in range(n_days):
      if n_days == 360:  
        d1 = d
        d2 = d
      else:
        d1 = random.randint(0,360)
        d2 = random.randint(0,360)      
      load_d1 = load.iloc[24*d1:24*(d1+1),:]
      wind_d2 = wind.iloc[24*d2:24*(d2+1),:]
      m,res = self.solve(load_d1,wind_d2,self.sta0(24),self.gen0(24),0)      
      for t in m.t:
        row = []
        for b in m.b:
          row.append(load_d1.iloc[t-1,b-1])
        for b in m.b:
          row.append(wind_d2.iloc[t-1,b-1])
        for l in m.l:
          flow = sum(self.ptdf[l-1][b-1]*m.inj[b,t].value for b in m.b)
          if abs(abs(flow) - self.lin['Pmax'][l-1]) <= 1e-6:
            if flow > 0:
              row.append(1)
            else:
              row.append(-1)
          else:
            row.append(0)        
        mat.append(row)  
      df = pd.DataFrame(mat)
      df.columns = ['d'+str(i) for i in m.b] + ['w'+str(i) for i in m.b] + ['l'+str(i) for i in m.l]  
      df.to_csv(data_file,index=False)
 
  # Draw figures with net demand and status of lines
  def draw_line(self,line,clusters):
      net_demand = pd.DataFrame(self.data.iloc[:,:self.n_bus].values-self.data.iloc[:,self.n_bus:2*self.n_bus].values)
      line_status = self.data.iloc[:,2*self.n_bus+line-1]
      ax = plt.gca()
      values = [-1,1,0]
      colors = ['r','b','g']
      for val,col in zip(values,colors):
        n_values = line_status.tolist().count(val)
        if n_values > 0:
          df = pd.DataFrame(KMeans(n_clusters=min(clusters,n_values), random_state=0).fit(net_demand.loc[line_status==val].values).cluster_centers_)
          df.T.loc[net_demand.mean().sort_values().index.tolist()].reset_index(drop=True).plot(color=col,legend=None,ax=ax)
      plt.title('line'+str(line))    
      plt.savefig('line'+str(line)+'.png')
      #plt.show()

  # Juanmi's filter
  def filter_data(self,data_file,output_file):
      n_data_filtered = 0 #counter of the number of filtered data points
      self.data = pd.read_csv(data_file, index_col=False)
      data_filter = self.data
      
      NetDemand = pd.DataFrame(self.data.iloc[:,:self.n_bus].as_matrix() - self.data.iloc[:,self.n_bus:2*self.n_bus].as_matrix())
      weights = self.ptdf
      # We associate aggregate generating capacity with each bus in the system
      gen_cap_per_node = pd.DataFrame(columns=['n', 'Pmax'])
      
      for n in range(self.n_bus):
        piece = self.gen[self.gen['n'] ==n]
        capacities = piece['Pmax']
        gen_cap_per_node = gen_cap_per_node.append({'n':n+1, 'Pmax':capacities.sum()}, ignore_index=True)
          
      for l,line in enumerate(weights):
        gen_cap_per_node['ptdf'] = pd.Series(line, index=gen_cap_per_node.index) 
        ptdf = gen_cap_per_node['ptdf']
        gen_cap_per_node = gen_cap_per_node.sort_values(['ptdf'], ascending = False)
        
        for i, nodal_demand in NetDemand.iterrows():
          flow_max = nodal_demand # Computation of the maximum power flow through line l
          flow_max = flow_max.multiply(ptdf)
          flow_max = flow_max.sum() # Contribution of net demand i to flow through line l
          flow_min = flow_max # Computation of the minimum power flow in line l 
          aggregate_demand = nodal_demand.sum() # Aggregate demand, i.e., total system demand for indivudual i
          total_generation = 0 # Total system generation from conventional units
          n = 0
          
          while total_generation + gen_cap_per_node['Pmax'].iloc[n] <= aggregate_demand: # We assume a well-dimensioned capacity-adequate system 
              total_generation = total_generation + gen_cap_per_node['Pmax'].iloc[n]
              flow_max = flow_max + gen_cap_per_node['ptdf'].iloc[n] * gen_cap_per_node['Pmax'].iloc[n]
              n += 1
          
          if aggregate_demand - total_generation > 0: # Renewable generation might be so high that the total net demand is negative   
              flow_max = flow_max + gen_cap_per_node['ptdf'].iloc[n] * (aggregate_demand - total_generation)
    
          total_generation = 0
          n = self.n_bus-1
          
          while total_generation + gen_cap_per_node['Pmax'].iloc[n] <= aggregate_demand:
              total_generation = total_generation + gen_cap_per_node['Pmax'].iloc[n]
              flow_min = flow_min + gen_cap_per_node['ptdf'].iloc[n] * gen_cap_per_node['Pmax'].iloc[n]
              n -= 1
          
          if aggregate_demand - total_generation > 0: # Renewable generation might be so high that the total net demand is negative   
              flow_min = flow_min + gen_cap_per_node['ptdf'].iloc[n] * (aggregate_demand - total_generation)  
        
          # Finally, we check whether line l is obviously uncongested for neta demand i
          
          if flow_max <= self.lin['Pmax'].iloc[l] and -self.lin['Pmax'].iloc[l] <= flow_min:
              # Line l is obviously congested for net demand i
              # We mark it in labels with a "3"
              data_filter.iloc[i,2*self.n_bus+l] = 3
              n_data_filtered += 1
              print('linea',l,'tiempo',i)
      
      print('Number of filtered data points: ', n_data_filtered)
      data_filter.to_csv(output_file,index=False)        
    
  # It generates the default matrix for the status of the lines (full of 2)
  def sta0(self,nper):
    sta = [[2 for i in range(self.n_lin)] for j in range(nper)]
    return pd.DataFrame(sta)

  # It generates the default matrix for the status of the generators (full of 0)
  def gen0(self,nper):
    gen = [[0 for i in range(nper)] for j in range(self.n_gen)]
    return pd.DataFrame(gen)
  
  # It generates the matrix for the status of the generators using the solution of the optimization model
  def gen1(self,model):
    gen = [[round(model.u[g,t].value) for t in model.t] for g in model.g]
    return pd.DataFrame(gen)

  # It solves the unit commitment optimization problem
  def solve_old(self,dem,win,lin,gen,fix):
    # dem: a data frame with the demand for all buses and time periods
    # win: a data frame with the wind generation for all buses and time periods
    # lin: status of transmission lines
    # gen: status of generating unit
    # fix: 1 (fix the commitment of generating units) and 0 (otherwise)
    #Define the model   
    m = pe.ConcreteModel()   
    #Define the sets        
    m.g = pe.Set(initialize=list(range(1,self.n_gen+1)),ordered=True) 
    m.l = pe.Set(initialize=list(range(1,self.n_lin+1)),ordered=True)
    m.b = pe.Set(initialize=list(range(1,self.n_bus+1)),ordered=True)  
    m.t = pe.Set(initialize=list(range(1,dem.shape[0]+1)),ordered=True)     
    #Define variables
    m.z = pe.Var()
    m.pro = pe.Var(m.g,m.t,within=pe.NonNegativeReals)
    m.u = pe.Var(m.g,m.t,within=pe.Binary)
    #m.shd = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.pos = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.neg = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.spl = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.ang = pe.Var(m.b,m.t)
    m.flw = pe.Var(m.l,m.t) 
    #Objective function
    def obj_rule(m):    
      return m.z
    m.obj = pe.Objective(rule=obj_rule)
    #Definition cost
    def cost_def_rule(m): 
      return m.z == sum(self.gen['c'][g-1]*m.pro[g,t] for g in m.g for t in m.t) + self.c_shed*sum(m.pos[b,t] + m.neg[b,t] for b in m.b for t in m.t)      
    m.cost_def = pe.Constraint(rule=cost_def_rule)
    #Energy balance
    def bal_rule(m,b,t):
      return sum(m.pro[g,t] for g in m.g if self.gen['n'][g-1] == b) + win.iloc[t-1,b-1] + m.pos[b,t] - m.neg[b,t] + sum(m.flw[l,t] for l in m.l if self.lin['to'][l-1] == b) == dem.iloc[t-1,b-1] + m.spl[b,t] + sum(m.flw[l,t] for l in m.l if self.lin['from'][l-1] == b)
    m.bal = pe.Constraint(m.b, m.t, rule=bal_rule)
    #Fix generation
    def fix_gen_rule(m,g,t):
        if fix == 1:
            return m.u[g,t] == gen.iloc[g-1,t-1]
        else:
            return pe.Constraint.Skip
    m.fix_gen = pe.Constraint(m.g,m.t,rule=fix_gen_rule)
    #Pos variables equal to 0 if commitment is not fix
    def pos0_rule(m,b,t):
        if fix == 0:
            return m.pos[b,t] ==0
        else:
            return pe.Constraint.Skip
    #m.pos0 = pe.Constraint(m.b,m.t,rule=pos0_rule)
    #Neg variables equal to 0 if commitment is not fix
    def neg0_rule(m,b,t):
        if fix == 0:
            return m.neg[b,t] ==0
        else:
            return pe.Constraint.Skip
    #m.neg0 = pe.Constraint(m.b,m.t,rule=neg0_rule)
    #Minimum generation
    def min_gen_rule(m,g,t):
      return m.pro[g,t] >= m.u[g,t]*self.gen['Pmin'][g-1]
    m.min_gen = pe.Constraint(m.g, m.t, rule=min_gen_rule)
    #Maximum generation
    def max_gen_rule(m,g,t):
      return m.pro[g,t] <= m.u[g,t]*self.gen['Pmax'][g-1]
    m.max_gen = pe.Constraint(m.g, m.t, rule=max_gen_rule)
    #Ramp up
    def ramp_up_rule(m,g,t):
        if t > 1:
            return m.pro[g,t] - m.pro[g,t-1] <= self.gen['Rup'][g-1]
        else:
            return pe.Constraint.Skip
    #m.ramp_up = pe.Constraint(m.g,m.t,rule=ramp_up_rule)
    #Ramp down
    def ramp_do_rule(m,g,t):
        if t > 1:
            return m.pro[g,t] - m.pro[g,t-1] >= -self.gen['Rdo'][g-1]
        else:
            return pe.Constraint.Skip
    #m.ramp_do = pe.Constraint(m.g,m.t,rule=ramp_do_rule)
    #Maximum spilage
    def max_spil_rule(m,b,t):
      return m.spl[b,t] <= win.iloc[t-1,b-1]
    m.max_spil = pe.Constraint(m.b, m.t, rule=max_spil_rule)
    #Maximum shedding
    #def max_shed_rule(m,b,t):
    #  return m.shd[b,t] <= dem.iloc[t-1,b-1]
    #m.max_shed = pe.Constraint(m.b, m.t, rule=max_shed_rule)
    #Power flow definition
    def flow_rule(m,l,t):
      return m.flw[l,t] == self.lin['x'][l-1]*(m.ang[self.lin['from'][l-1],t] - m.ang[self.lin['to'][l-1],t])
    m.flow = pe.Constraint(m.l, m.t, rule=flow_rule)
    #Max power flow
    def max_flow_rule(m,l,t):
        #if lin.iloc[t-1,l-1] == -1 :
        #    return m.flw[l,t] <= -self.lin['Pmax'][l-1]
        if lin.iloc[t-1,l-1] == 0 :
            return pe.Constraint.Skip
        else:
            return m.flw[l,t] <= self.lin['Pmax'][l-1]
    m.max_flow = pe.Constraint(m.l, m.t, rule=max_flow_rule)
    #Min power flow
    def min_flow_rule(m,l,t):
        #if lin.iloc[t-1,l-1] == 1 :
        #   return m.flw[l,t] >= self.lin['Pmax'][l-1]
        if lin.iloc[t-1,l-1] == 0 :
            return pe.Constraint.Skip
        else:
            return m.flw[l,t] >= -self.lin['Pmax'][l-1]         
    m.min_flow = pe.Constraint(m.l, m.t, rule=min_flow_rule)
    #We solve the optimization problem
    opt = pe.SolverFactory('cplex',symbolic_solver_labels=True,tee=True)
    opt.options['threads'] = 1   
    opt.options['mipgap'] = 1e-9
    #opt.options['mip_strategy_search'] = 2
    res = opt.solve(m,symbolic_solver_labels=True,tee=True)   
    if str(res.solver.termination_condition) == 'optimal':
      m.hour_dem = pd.DataFrame(dem).sum(axis=1).to_list()
      m.hour_shd = [sum([m.pos[b,t].value+m.neg[b,t].value for b in m.b]) for t in m.t]
      m.max_hour_shd = max([100*m.hour_shd[i]/m.hour_dem[i] for i in range(len(m.hour_dem))])
      m.day_dem = sum(m.hour_dem)
      m.day_shd = sum(m.hour_shd)
      m.cost = m.z.value - self.c_shed*m.day_shd
    print(res['Solver'][0])
    return m,res
  
  # It solves the unit commitment optimization problem
  def solve(self,dem,win,lin,gen,fix):
    # dem: a data frame with the demand for all buses and time periods
    # win: a data frame with the wind generation for all buses and time periods
    # lin: status of transmission lines
    # gen: status of generating unit
    # fix: 1 (fix the commitment of generating units) and 0 (otherwise)
    #Define the model   
    m = pe.ConcreteModel()   
    #Define the sets        
    m.g = pe.Set(initialize=list(range(1,self.n_gen+1)),ordered=True) 
    m.l = pe.Set(initialize=list(range(1,self.n_lin+1)),ordered=True)
    m.b = pe.Set(initialize=list(range(1,self.n_bus+1)),ordered=True)  
    m.t = pe.Set(initialize=list(range(1,dem.shape[0]+1)),ordered=True)     
    #Define variables
    m.z = pe.Var()
    m.pro = pe.Var(m.g,m.t,within=pe.NonNegativeReals)
    m.u = pe.Var(m.g,m.t,within=pe.Binary)
    m.pos = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.neg = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.spl = pe.Var(m.b,m.t,within=pe.NonNegativeReals)
    m.inj = pe.Var(m.b,m.t)
    #Objective function
    def obj_rule(m):    
      return m.z
    m.obj = pe.Objective(rule=obj_rule)
    #Definition cost
    def cost_def_rule(m): 
      return m.z == sum(self.gen['c'][g-1]*m.pro[g,t] for g in m.g for t in m.t) + self.c_shed*sum(m.pos[b,t] + m.neg[b,t] for b in m.b for t in m.t)      
    m.cost_def = pe.Constraint(rule=cost_def_rule)
    #Injected power
    def inj_power_rule(m,b,t):
      return m.inj[b,t] == sum(m.pro[g,t] for g in m.g if self.gen['n'][g-1] == b) + win.iloc[t-1,b-1] + m.pos[b,t] - m.neg[b,t] - dem.iloc[t-1,b-1] - m.spl[b,t] 
    m.ing_power = pe.Constraint(m.b, m.t, rule=inj_power_rule)
    #Power balance
    def bal_rule(m,t):
      return sum(m.inj[b,t] for b in m.b) == 0
    m.bal = pe.Constraint(m.t,rule=bal_rule)
    #Fix generation
    def fix_gen_rule(m,g,t):
        if fix == 1:
            return m.u[g,t] == gen.iloc[g-1,t-1]
        else:
            return pe.Constraint.Skip
    m.fix_gen = pe.Constraint(m.g,m.t,rule=fix_gen_rule)
    #Pos variables equal to 0 if commitment is not fix
    def pos0_rule(m,b,t):
        if fix == 0:
            return m.pos[b,t] ==0
        else:
            return pe.Constraint.Skip
    m.pos0 = pe.Constraint(m.b,m.t,rule=pos0_rule)
    #Neg variables equal to 0 if commitment is not fix
    def neg0_rule(m,b,t):
        if fix == 0:
            return m.neg[b,t] ==0
        else:
            return pe.Constraint.Skip
    m.neg0 = pe.Constraint(m.b,m.t,rule=neg0_rule)
    #Minimum generation
    def min_gen_rule(m,g,t):
      return m.pro[g,t] >= m.u[g,t]*self.gen['Pmin'][g-1]
    m.min_gen = pe.Constraint(m.g, m.t, rule=min_gen_rule)
    #Maximum generation
    def max_gen_rule(m,g,t):
      return m.pro[g,t] <= m.u[g,t]*self.gen['Pmax'][g-1]
    m.max_gen = pe.Constraint(m.g, m.t, rule=max_gen_rule)
    #Maximum spilage
    def max_spil_rule(m,b,t):
      return m.spl[b,t] <= win.iloc[t-1,b-1]
    m.max_spil = pe.Constraint(m.b, m.t, rule=max_spil_rule)
    #Max power flow
    def max_flow_rule(m,l,t):
        if lin.iloc[t-1,l-1] == 0 :
            return pe.Constraint.Skip
        else:
            return sum(self.ptdf[l-1][b-1]*m.inj[b,t] for b in m.b)  <= self.lin['Pmax'][l-1]
    m.max_flow = pe.Constraint(m.l, m.t, rule=max_flow_rule)
    #Min power flow
    def min_flow_rule(m,l,t):
        if lin.iloc[t-1,l-1] == 0 :
            return pe.Constraint.Skip
        else:
            return sum(self.ptdf[l-1][b-1]*m.inj[b,t] for b in m.b) >= -self.lin['Pmax'][l-1]         
    m.min_flow = pe.Constraint(m.l, m.t, rule=min_flow_rule)
    #We solve the optimization problem
    opt = pe.SolverFactory('cplex',symbolic_solver_labels=True,tee=True)
    opt.options['threads'] = 1   
    opt.options['mipgap'] = 1e-9
    #opt.options['mip_strategy_search'] = 2
    res = opt.solve(m,symbolic_solver_labels=True,tee=True)   
    if str(res.solver.termination_condition) == 'optimal':
      m.hour_dem = pd.DataFrame(dem).sum(axis=1).to_list()
      m.hour_shd = [sum([m.pos[b,t].value+m.neg[b,t].value for b in m.b]) for t in m.t]
      m.max_hour_shd = max([100*m.hour_shd[i]/m.hour_dem[i] for i in range(len(m.hour_dem))])
      m.day_dem = sum(m.hour_dem)
      m.day_shd = sum(m.hour_shd)
      m.cost = m.z.value - self.c_shed*m.day_shd
    print(res['Solver'][0])
    return m,res

  # It provides the maximum or minimum flow for a given line
  def max_flow_roald(self,ini_train,end_train,sense,line,alpha=0):
    #Compute of maximum and minimum net demand of the train data  
    net_dem = pd.DataFrame(self.data.iloc[24*ini_train:24*end_train,:self.n_bus].values-self.data.iloc[24*ini_train:24*end_train,self.n_bus:2*self.n_bus].values)
    max_net_dem = net_dem.quantile(q=alpha)
    min_net_dem = net_dem.quantile(q=1-alpha)
    #Define the model   
    m = pe.ConcreteModel()   
    #Define the sets        
    m.g = pe.Set(initialize=list(range(1,self.n_gen+1)),ordered=True) 
    m.l = pe.Set(initialize=list(range(1,self.n_lin+1)),ordered=True)
    m.b = pe.Set(initialize=list(range(1,self.n_bus+1)),ordered=True)  
    #Define variables
    m.z = pe.Var()
    m.pro = pe.Var(m.g,within=pe.NonNegativeReals)
    m.ang = pe.Var(m.b)
    m.flw = pe.Var(m.l) 
    m.dem = pe.Var(m.b)
    #Objective function
    def obj_rule(m):    
      return m.z
    m.obj = pe.Objective(rule=obj_rule)
    #Definition cost
    def cost_def_rule(m): 
      return m.z == sense*sum(m.flw[l] for l in m.l if l==line)
    m.cost_def = pe.Constraint(rule=cost_def_rule)
    #Energy balance
    def bal_rule(m,b):
      return sum(m.pro[g] for g in m.g if self.gen['n'][g-1] == b) + sum(m.flw[l] for l in m.l if self.lin['to'][l-1] == b) == m.dem[b] + sum(m.flw[l] for l in m.l if self.lin['from'][l-1] == b)
    m.bal = pe.Constraint(m.b, rule=bal_rule)
    #Maximum net demand
    def max_net_rule(m,b):
        return m.dem[b] <= max_net_dem[b-1]
    m.max_net = pe.Constraint(m.b, rule=max_net_rule)
    #Minimum net demand
    def min_net_rule(m,b):
        return m.dem[b] >= min_net_dem[b-1]
    m.min_net = pe.Constraint(m.b, rule=min_net_rule)
    #Minimum generation
    def min_gen_rule(m,g):
      return m.pro[g] >= 0
    m.min_gen = pe.Constraint(m.g, rule=min_gen_rule)
    #Maximum generation
    def max_gen_rule(m,g):
      return m.pro[g] <= self.gen['Pmax'][g-1]
    m.max_gen = pe.Constraint(m.g, rule=max_gen_rule)
    #Power flow definition
    def flow_rule(m,l):
      return m.flw[l] == self.lin['x'][l-1]*(m.ang[self.lin['from'][l-1]] - m.ang[self.lin['to'][l-1]])
    m.flow = pe.Constraint(m.l, rule=flow_rule)
    #Max power flow
    def max_flow_rule(m,l):
        return m.flw[l] <= self.lin['Pmax'][l-1]
    m.max_flow = pe.Constraint(m.l, rule=max_flow_rule)
    #Min power flow
    def min_flow_rule(m,l):
        return m.flw[l] >= -self.lin['Pmax'][l-1]         
    m.min_flow = pe.Constraint(m.l, rule=min_flow_rule)
    #We solve the optimization problem
    opt = pe.SolverFactory('cplex',symbolic_solver_labels=True,tee=True)
    opt.options['threads'] = 1   
    opt.options['mipgap'] = 1e-9
    res = opt.solve(m,symbolic_solver_labels=True,tee=True)   
    print(res['Solver'][0])
    #f = open('borrar_line.csv', 'a')
    if self.lin['Pmax'][line-1] == abs(m.z.value):
        #f.write('Line '+str(line)+' has reached its capacity '+str(sense*m.z.value)+'\n')
        status = False
    else:
        #f.write('Line '+str(line)+' has not reached its capacity '+str(sense*m.z.value)+'\n')
        status = True
    #f.close()
    return status

  # It provides the max flow through all lines as explain in Zhai 2010 
  def max_flow_zhai(self):
    #Define the model   
    m = pe.ConcreteModel()   
    #Define the sets        
    m.g = pe.Set(initialize=list(range(1,self.n_gen+1)),ordered=True) 
    m.l = pe.Set(initialize=list(range(1,self.n_lin+1)),ordered=True)
    m.b = pe.Set(initialize=list(range(1,self.n_bus+1)),ordered=True)  
    #Define parameters
    m.dem = pe.Param(m.b,mutable=True,initialize=list2dict([0 for b in m.b]))
    m.win = pe.Param(m.b,mutable=True,initialize=list2dict([0 for b in m.b]))
    m.line = pe.Param(m.l,mutable=True,initialize=list2dict([0 for l in m.l]))
    m.sense = pe.Param(mutable=True,initialize=1)
    #Define variables
    m.z = pe.Var()
    m.pro = pe.Var(m.g,within=pe.NonNegativeReals)
    m.spl = pe.Var(m.b,within=pe.NonNegativeReals)
    m.ang = pe.Var(m.b)
    m.flw = pe.Var(m.l) 
    #Objective function
    def obj_rule(m):    
      return m.z
    m.obj = pe.Objective(rule=obj_rule)
    #Definition cost
    def cost_def_rule(m): 
      return m.z == m.sense*sum(m.line[l]*m.flw[l] for l in m.l)       
    m.cost_def = pe.Constraint(rule=cost_def_rule)
    #Energy balance
    def bal_rule(m,b):
      return sum(m.pro[g] for g in m.g if self.gen['n'][g-1] == b) + m.win[b] + sum(m.flw[l] for l in m.l if self.lin['to'][l-1] == b) == m.dem[b] + m.spl[b] + sum(m.flw[l] for l in m.l if self.lin['from'][l-1] == b)
    m.bal = pe.Constraint(m.b, rule=bal_rule)
    #Maximum generation
    def max_gen_rule(m,g):
      return m.pro[g] <= self.gen['Pmax'][g-1]
    m.max_gen = pe.Constraint(m.g, rule=max_gen_rule)
    #Maximum spilage
    def max_spil_rule(m,b):
      return m.spl[b] <= m.win[b]
    m.max_spil = pe.Constraint(m.b, rule=max_spil_rule)
    #Power flow definition
    def flow_rule(m,l):
      return m.flw[l] == self.lin['x'][l-1]*(m.ang[self.lin['from'][l-1]] - m.ang[self.lin['to'][l-1]])
    m.flow = pe.Constraint(m.l, rule=flow_rule)
    #We solve the optimization problem
    #opt = pe.SolverFactory('cplex',symbolic_solver_labels=True,tee=True)
    #opt.options['threads'] = 1   
    #res = opt.solve(m,symbolic_solver_labels=True,tee=True)   
    #print(res['Solver'][0])
    #if self.lin['Pmax'][line-1] == abs(m.z.value):
    #    status = False
    #else:
    #    status = True
    return m

# It provides the time periods for which a given line is certainly not congested in a given flow direction ("sense")
  def oracle(self,ini_day,end_day,sense,line,alpha=0, beta=1):
    #ini_day, end_day: The congestion status of line will be determined for the time periods in day "ini_day" (first day being 0) and day "end_day" (excluding periods in this)
    #sense: indicates the direction in which the maximum power flow is computed {1, -1}
    #line: indicates the line under consideration
    #alpha: indicates the maximum in per unit of shed load that is allowed
    #beta: indicates the maximum in per unit of wind spillage that is permitted
    
    #Retrieving demand and wind generation data: 
    dem = pd.DataFrame(self.data.iloc[24*ini_day:24*end_day,:self.n_bus].values)
    wind = pd.DataFrame(self.data.iloc[24*ini_day:24*end_day,self.n_bus:2*self.n_bus].values)
    
    #Define the model   
    m = pe.ConcreteModel() 
    
    #Define the sets        
    m.g = pe.Set(initialize=list(range(1,self.n_gen+1)),ordered=True) 
    m.l = pe.Set(initialize=list(range(1,self.n_lin+1)),ordered=True)
    m.b = pe.Set(initialize=list(range(1,self.n_bus+1)),ordered=True)  
    m.t = pe.Set(initialize=list(range(1,24*(end_day-ini_day)+1)),ordered=True)
    
    #Define variables
    m.z = pe.Var()
    m.pro = pe.Var(m.g, m.t,within=pe.NonNegativeReals)
    m.ang = pe.Var(m.b, m.t)
    m.flw = pe.Var(m.l, m.t) 
    m.ls = pe.Var(m.b, m.t, within=pe.NonNegativeReals) #load shedding
    m.ws = pe.Var(m.b, m.t, within=pe.NonNegativeReals) #wind spillage

    #Objective function
    def obj_rule(m):    
      return m.z
    m.obj = pe.Objective(rule=obj_rule)
    
    #Definition cost
    def cost_def_rule(m): 
      return m.z == sense*sum(sum(m.flw[l,t] for l in m.l if l==line) for t in m.t)
    m.cost_def = pe.Constraint(rule=cost_def_rule)
    
    #Energy balance
    def bal_rule(m,b,t):
      return sum(m.pro[g,t] for g in m.g if self.gen['n'][g-1] == b) - m.ws[b,t] + sum(m.flw[l,t] for l in m.l if self.lin['to'][l-1] == b) == dem.iloc[t-1,b-1] - m.ls[b,t] + sum(m.flw[l,t] for l in m.l if self.lin['from'][l-1] == b)
    m.bal = pe.Constraint(m.b, m.t, rule=bal_rule)
    
   #Maximum load shedding per node and per period
    def max_load_shed_n(m,b,t):
        return m.ls[b,t] <= dem.iloc[t-1,b-1]
    m.max_ls_n = pe.Constraint(m.b, m.t, rule=max_load_shed_n)
    
    #Maximum total load shedding in the system per period
    def max_load_shed(m,t):
        return sum(m.ls[b,t] for b in m.b) <= alpha*dem.sum(axis=1).iloc[t-1]
    m.max_ls = pe.Constraint(m.t, rule=max_load_shed)

    #Maximum wind spillage per node and per period
    def max_wind_spill_n(m,b,t):
        return m.ws[b,t] <= wind.iloc[t-1,b-1]
    m.max_ws_n = pe.Constraint(m.b, m.t, rule=max_wind_spill_n)
    
    #Maximum total wind spillage in the system per period
    def max_wind_spill(m,t):
        return sum(m.ws[b,t] for b in m.b) <= beta*wind.sum(axis=1).iloc[t-1]
    m.max_ws = pe.Constraint(m.t, rule=max_wind_spill)
    
    #Minimum generation
    def min_gen_rule(m,g,t):
      return m.pro[g,t] >= 0
    m.min_gen = pe.Constraint(m.g,m.t, rule=min_gen_rule)
    
    #Maximum generation
    def max_gen_rule(m,g,t):
      return m.pro[g,t] <= self.gen['Pmax'][g-1]
    m.max_gen = pe.Constraint(m.g,m.t, rule=max_gen_rule)
    
    #Power flow definition
    def flow_rule(m,l,t):
      return m.flw[l,t] == self.lin['x'][l-1]*(m.ang[self.lin['from'][l-1],t] - m.ang[self.lin['to'][l-1],t]) #### IS THIS REACTANCE OR SUSCEPTANCE???
    m.flow = pe.Constraint(m.l,m.t, rule=flow_rule)
    
    #Max power flow
    def max_flow_rule(m,l,t):
        return m.flw[l,t] <= self.lin['Pmax'][l-1]
    m.max_flow = pe.Constraint(m.l,m.t, rule=max_flow_rule)
    
    #Min power flow
    def min_flow_rule(m,l,t):
        return m.flw[l,t] >= -self.lin['Pmax'][l-1]         
    m.min_flow = pe.Constraint(m.l,m.t, rule=min_flow_rule)
    
    #We solve the optimization problem
    opt = pe.SolverFactory('cplex',symbolic_solver_labels=True,tee=True)
    opt.options['threads'] = 1   
    opt.options['mipgap'] = 1e-9
    opt.options['mip_tolerances_integrality'] = 1e-15
    
    res = opt.solve(m,symbolic_solver_labels=True,tee=True)   
    print(res['Solver'][0])
    #f = open('borrar_line.csv', 'a')
    
    status = []
    if str(res.solver.termination_condition) == 'optimal':
      for t in m.t:
        if self.lin['Pmax'][line-1] == abs(m.flw[line,t].value):
          #f.write('Line '+str(line)+' has reached its capacity '+str(sense*m.z.value)+'\n')
          status.insert(t-1, False)
        else:
          #f.write('Line '+str(line)+' has not reached its capacity '+str(sense*m.z.value)+'\n')
          status.insert(t-1, True)
          #f.close()
    else: 
      for t in m.t:
          status.insert(t-1, False)
    return status

 # Juanmi's filter: Second Version
  def filter_data_improved(self, ini_day, end_day, line, writing_file, alpha=0, beta=1):
      n_data_filtered = 0 #counter of the number of filtered data points      
      data_filter = self.data.iloc[:, 2*self.n_bus+line-1].tolist()
      
      status1 = []
      status2 = []         
#      for line in range(0,self.n_lin): #iteration over the lines
      for d in range(ini_day, end_day): #iteration over days
        status1.extend(self.oracle(d,d+1,1,line, alpha, beta))
        status2.extend(self.oracle(d,d+1,-1,line, alpha, beta))
      
      status = status1 and status2 
      for i,j in enumerate(status):
         if j:
            # Line l is obviously congested for net demand i
            # We mark it in labels with a "3"
            data_filter[i] = 3
            n_data_filtered += 1
 #          print('linea',l,'tiempo',i)          
      
      data_filter = ",".join([str(x) for x in data_filter])
      data_filter = "l"+str(line)+","+ data_filter
      f = open(writing_file, 'a')
      f.write(data_filter +'\n')
      f.close()
 #     return data_filter 
 #     print('Number of filtered data points: ', n_data_filtered)
 #     data_filter.to_csv(output_file,index=False)        
              
    

  
  # function to select the training and test data and weights of features
  def learning_test_data(self,
                         ini_train,
                         end_train,
                         ini_test,
                         end_test,
                         periods = 1,
                         net_demand=False,
                         weight_ptdf=False):
    self.ini_train = ini_train
    self.end_train = end_train
    self.ini_test = ini_test
    self.end_test = end_test
    self.period = periods
    self.net_demand = net_demand
    self.weight_ptdf = weight_ptdf

    if net_demand:
      x_train = pd.DataFrame(self.data.iloc[24*ini_train:24*end_train,:self.n_bus].values-self.data.iloc[24*ini_train:24*end_train,self.n_bus:2*self.n_bus].values)  
      x_test = pd.DataFrame(self.data.iloc[24*ini_test:24*end_test,:self.n_bus].values-self.data.iloc[24*ini_test:24*end_test,self.n_bus:2*self.n_bus].values)  
      if weight_ptdf:
          weights = self.ptdf #ToDo: discuss this point with Salva
          
      else:
          weights = [1 for i in range(self.n_bus)]
    else:  
      x_train = self.data.iloc[24*ini_train:24*end_train,:2*self.n_bus]
      x_test = self.data.iloc[24*ini_test:24*end_test,:2*self.n_bus].reset_index().iloc[:,1:]
      if weight_ptdf:
          weights = [self.ptdf[i] + self.ptdf[i] for i in range(self.n_lin)]
      else:
          weights = [1 for i in range(2*self.n_bus)] 
    y_train = self.data.iloc[24*ini_train:24*end_train,2*self.n_bus:]    
    y_test = self.data.iloc[24*ini_test:24*end_test,2*self.n_bus:]
    if periods==3:
      weights = [weights[i] + weights[i] + weights[i] for i in range(len(weights))]
      mat = []
      for index, row in x_train.iterrows():
        if (index+24)%24==0:
          mat.append(x_train.iloc[index,:].to_list()+x_train.iloc[index,:].to_list()+x_train.iloc[index+1,:].to_list())
        elif (index+1)%24==0:
          mat.append(x_train.iloc[index-1,:].to_list()+x_train.iloc[index,:].to_list()+x_train.iloc[index,:].to_list())
        else:
          mat.append(x_train.iloc[index-1,:].to_list()+x_train.iloc[index,:].to_list()+x_train.iloc[index+1,:].to_list())
      x_train = pd.DataFrame(mat)
      mat = []
      for index, row in x_test.iterrows():
        if (index+24)%24==0:
          mat.append(x_test.iloc[index,:].to_list()+x_test.iloc[index,:].to_list()+x_test.iloc[index+1,:].to_list())
        elif (index+1)%24==0:
          mat.append(x_test.iloc[index-1,:].to_list()+x_test.iloc[index,:].to_list()+x_test.iloc[index,:].to_list())
        else:
          mat.append(x_test.iloc[index-1,:].to_list()+x_test.iloc[index,:].to_list()+x_test.iloc[index+1,:].to_list())
      x_test = pd.DataFrame(mat)
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.weights = weights
    
    #Discuss this change with Salva
    return self

  def learn_line(self,
                 method,
                 level=1,
                 net_demand = True,
                 weight_ptdf = True,
                 weights_values = np.abs([[1, 1, 1],[1, 1, 1],[1, 1, 1]]),
                 SVM_regularization_parameter_grid = [10**range_element for range_element in range(0, 1)]):
    self.method = method  
    self.level = level
    mat_pred = []
    mat_prob = []
    score = []
    mat_score = [[0,0,0],[0,0,0],[0,0,0]]
    mat_score_trust = [[0,0,0],[0,0,0],[0,0,0]]
    num_total = 0
    num_trust = 0
    num_right = 0
    
    #Method 11: illustrative m2svm optimization
    if method == 'illustrative_m2svm_optimization':
        if net_demand:
            x_train = pd.DataFrame(self.data.iloc[self.ini_train:self.end_train,:self.n_bus].values-self.data.iloc[self.ini_train:self.end_train,self.n_bus:2*self.n_bus].values)  
            x_test = pd.DataFrame(self.data.iloc[self.ini_test:self.end_test,:self.n_bus].values-self.data.iloc[self.ini_test:self.end_test,self.n_bus:2*self.n_bus].values)  
            
            y_train = self.data.iloc[self.ini_train:self.end_train,2*self.n_bus:]    
            y_test = self.data.iloc[self.ini_test:self.end_test,2*self.n_bus:]
            if weight_ptdf:
                weights = np.abs(self.ptdf) #ToDo: discuss this point with Salva.
                #weights = np.abs([[1, 0, 0],[0.5, 0, 0],[1, 0, 0]])
            else:
                weights = [[1 for i in range(self.n_bus)] for line in range(self.n_lin)]
                #weights = weights_values
        else:  
            x_train = self.data.iloc[self.ini_train:self.end_train,:2*self.n_bus]
            x_test = self.data.iloc[self.ini_test:self.end_test,:2*self.n_bus].reset_index().iloc[:,1:]
            
            y_train = self.data.iloc[self.ini_train:self.end_train,self.n_bus:]    
            y_test = self.data.iloc[self.ini_test:self.end_test,self.n_bus:]
            if weight_ptdf:
                weights = [self.ptdf[i] + self.ptdf[i] for i in range(self.n_lin)]
            else:
                weights = [[1 for i in range(2*self.n_bus)] for line in range(self.n_lin)]
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.weights = weights
        indexes_individuals_sample_test = pd.DataFrame(data = np.array(range(len(self.x_train) + 1, len(self.x_train) + len(self.x_test) + 1)),
                                                       index = range(len(self.x_train) + 1, len(self.x_train) + len(self.x_test) + 1),
                                                       columns = ['time_period'])
        indexes_individuals_sample_training = pd.DataFrame(data = np.array(range(1, len(self.x_train) + 1)),
                                                           index = range(1, len(self.x_train) + 1),
                                                           columns = ['time_period'])
        best_results_tune_parameters_grid = {}
        seed_sampling = 1309
        seed_shuffle = 1615
        sample_names = ['training_1',
                      'training_2',
                      'validation',
                      'test']
        number_of_samples_except_testing = 3
        number_of_lines = self.y_train.shape[1]
        output_sampling = smp.sampling_method(indexes_individuals_sample_training = indexes_individuals_sample_training,
                                          indexes_individuals_sample_test = indexes_individuals_sample_test,
                                          number_of_lines = number_of_lines,
                                          seed_sampling = seed_sampling,
                                          seed_shuffle = seed_shuffle,
                                          sample_names = sample_names,
                                          number_of_samples_except_testing = number_of_samples_except_testing)
        samples = output_sampling['indexes_individuals_all_samples_and_lines']
        correspondence_time_period_line_all_samples = output_sampling['correspondence_time_period_line_all_samples']
        
        individuals_training_1 = list(range(1, 13))
        individuals_training_2 = [13, 16, 17, 20, 21, 23, 26, 27, 28, 31]
        individuals_validation = [14, 15, 18, 19, 22, 24, 25, 29, 30]
        for line in range(number_of_lines):
            samples[line]['training_1'] = individuals_training_1
            samples[line]['training_2'] = individuals_training_2
            samples[line]['validation'] = individuals_validation
        
        line_to_learn = 2
        correspondence_time_period_line_all_samples['training_1'] = pd.DataFrame(data = {'time_period': individuals_training_1,
                                                                                          'line': [line_to_learn]*len(individuals_training_1)},
                                                                                 index = range(1, len(individuals_training_1) + 1))
        correspondence_time_period_line_all_samples['training_2'] = pd.DataFrame(data = {'time_period': individuals_training_2,
                                                                                          'line': [line_to_learn]*len(individuals_training_2)},
                                                                                 index = range(1, len(individuals_training_2) + 1))
        correspondence_time_period_line_all_samples['validation'] = pd.DataFrame(data = {'time_period': individuals_validation,
                                                                                          'line': [line_to_learn]*len(individuals_validation)},
                                                                                 index = range(1, len(individuals_validation) + 1))
        
        
        data_train_normalized = nm.normalize_data(data = self.x_train)
        data_test_normalized = nm.normalize_data(data = self.x_test)
        
        
        
        seed_random_prediction_values = 1451
        lowest_label_value = -1
        highest_label_value = 1
        sample_to_get_best_parameters = sample_names[2]
        maximum_number_iterations_alternating_approach = 1
        threshold_difference_objective_values_second_step = 1e-5
        default_difference_objective_values_second_step = 1e5
        seed_initialize_parameters = 1133
        number_of_nodes = self.n_bus
        bounds_weights = {'lower_bound': 0,
                         'upper_bound_initial_solution': 10}
        label_values = [-1, 0, 1]
        new_label_values = [1, 2, 3]
        if net_demand:
            number_of_renewable_energy = 0
        else:
            number_of_renewable_energy = 1
        maximum_number_iterations_multistart = 5
        perturbation_multistart_variables = {'weights': 1}
        seed_multistart = 1219
        default_new_objective_value_second_step = 1e3
        beggining_file_name_to_save_results = 'results_by_line_'
        folder_results_msvm = 'results_msvm/'
        if not (os.path.isdir('./' + folder_results_msvm)):
            os.mkdir(folder_results_msvm)
        csv_file_name = 'results_all_lines'
      
        if weight_ptdf:
            approach = 'ptdf'
        else:
            approach = 'random'
        csv_file = folder_results_msvm + csv_file_name +'_' + approach + '.csv'
        ######################################################################################
        # This code is not necessary in the solvers comparison
#        file_to_write = open(csv_file, 'a')
#        file_to_write.write('data_file' + ',' + 'line' + ',' +"%zeros" +',' +"%ones" +',' +"%minus_ones" +',' + 'approach' + ',' + '% total accuracy' + ',' + 'SVM reg param' + ',' +'weights'+'\n')
#        file_to_write.close()
#        
#        pltu.plot_individuals_samples(sample_by_line = samples[line],
#                                      sample_names = ['training_1', 'training_2', 'validation'],
#                                      data = data_train_normalized,
#                                      label = y_train.copy().iloc[:,line_to_learn - 1],
#                                      label_values = label_values,
#                                      folder_results_msvm = folder_results_msvm)
        ######################################################################################
        for line in range(1, 2):
            print("The congestion of line %d is learnt with the multiclass SVM method" % (line + 1))
            data = pd.concat([data_train_normalized, data_test_normalized])
            data.index = range(1, len(data) + 1)
            label = pd.concat([self.y_train, self.y_test])
            label.index = range(1, len(label) + 1)
          
            percentage_individuals_of_label_zero = 100*len(label.iloc[:,line][label.iloc[:,line] == 0])/len(label.iloc[:,line])
            percentage_individuals_of_label_one = 100*len(label.iloc[:,line][label.iloc[:,line] == 1])/len(label.iloc[:,line])
            percentage_individuals_of_label_minus_one = 100*len(label.iloc[:,line][label.iloc[:,line] == -1])/len(label.iloc[:,line])
            
            
            if weight_ptdf:
               initial_weights = np.absolute(np.array(self.ptdf[line]))
            else:
               initial_weights = [None]
            pdb.set_trace()
            best_results_tune_parameters_grid[line] = ptg.tune_parameters_grid(SVM_regularization_parameter_grid = SVM_regularization_parameter_grid,
                                                                           sample_by_line = samples[line],
                                                                           label = label,
                                                                           line = line,
                                                                           sample_names = sample_names,
                                                                           seed_random_prediction_values = seed_random_prediction_values,
                                                                           lowest_label_value = lowest_label_value,
                                                                           highest_label_value = highest_label_value,
                                                                           sample_to_get_best_parameters = sample_to_get_best_parameters,
                                                                           maximum_number_iterations_alternating_approach = maximum_number_iterations_alternating_approach,
                                                                           threshold_difference_objective_values_second_step = threshold_difference_objective_values_second_step,
                                                                           default_difference_objective_values_second_step = default_difference_objective_values_second_step,
                                                                           seed_initialize_parameters = seed_initialize_parameters,
                                                                           number_of_nodes = number_of_nodes,
                                                                           bounds_weights = bounds_weights,
                                                                           label_values = label_values,
                                                                           new_label_values = new_label_values,
                                                                           data = data,
                                                                           number_of_renewable_energy = number_of_renewable_energy,
                                                                           correspondence_time_period_line_all_samples = correspondence_time_period_line_all_samples,
                                                                           maximum_number_iterations_multistart = maximum_number_iterations_multistart,
                                                                           perturbation_multistart_variables = perturbation_multistart_variables,
                                                                           seed_multistart = seed_multistart,
                                                                           default_new_objective_value_second_step = default_new_objective_value_second_step,
                                                                           initial_weights = initial_weights)
            pdb.set_trace()
            file_name_to_save_results = folder_results_msvm + beggining_file_name_to_save_results + str(line + 1) +'_' + approach+'.pydata'          
            file_to_save = open(file_name_to_save_results, 'wb')
            pickle.dump(best_results_tune_parameters_grid[line], file_to_save)
            file_to_save.close()
            score.append(best_results_tune_parameters_grid[line]['accuracy_all_samples']['test'])
            mat_pred.append((best_results_tune_parameters_grid[line]['prediction_all_samples']['test']).tolist())
            mat_prob.append((-0.5*np.ones(shape = len(y_test_line),dtype = float)).tolist())
          
            file_to_read = open(file_name_to_save_results, 'rb')
            results_to_save = pickle.load(file_to_read)
          
                    
            file_to_write = open(csv_file, 'a')
            file_to_write.write(self.data_file + ',' + str((line + 1)) + ','+ str(percentage_individuals_of_label_zero +approach) + ',' +str(percentage_individuals_of_label_one) + ','+str(percentage_individuals_of_label_minus_one)+',' + str(results_to_save['accuracy_all_samples']['test']) + ',' +str(best_results_tune_parameters_grid[line]['best_SVM_regularization_parameter']) +',' +','.join(map(str, list(results_to_save['weights'][0].values)))+'\n')
            file_to_write.close()
        
      
        file_name_all_results = folder_results_msvm + 'all_results'+ '_' + approach + '.pydata'
        file_to_save = open(file_name_all_results, 'wb')
        pickle.dump((best_results_tune_parameters_grid, score, mat_pred, mat_prob), file_to_save)
        file_to_save.close()
    
    # End if for methods
    mat_true = self.y_test.T.values.tolist()    
    for line in range(len(mat_pred)):
      for time in range(len(mat_pred[0])):
        mat_score[int(mat_true[line][time])+1][int(mat_pred[line][time])+1] += 1  
        if mat_prob[line][time] < level:
          mat_pred[line][time] = 2  
        else:  
          mat_score_trust[int(mat_true[line][time])+1][int(mat_pred[line][time])+1] += 1  
    self.line_forecast = pd.DataFrame(mat_pred).T
    #self.score = 100*sum(score)/len(score)
    self.mat_score = mat_score      
    self.mat_score_trust = mat_score_trust
 
  # Write mat_score to a text file
  def write_mat_score(self,text_file):
    f = open(text_file, 'a')
    f.write(self.data_file+','+self.method+','+str(self.level)+','+str(self.mat_score_trust[0][0])+','+str(self.mat_score_trust[0][1])+','+str(self.mat_score_trust[0][2])+','+str(self.mat_score_trust[1][0])+','+str(self.mat_score_trust[1][1])+','+str(self.mat_score_trust[1][2])+','+str(self.mat_score_trust[2][0])+','+str(self.mat_score_trust[2][1])+','+str(self.mat_score_trust[2][2])+'\n')
    f.close()

  # Compute the load shedding
  def compute_shd(self,data_file,ini,end):
      data = pd.read_csv(data_file, index_col=False)
      shd = 0
      dem = 0
      for i,t in enumerate(range(ini,end)):
        dem_day = data.iloc[24*t:24*(t+1),:self.n_bus]
        win_day = data.iloc[24*t:24*(t+1),self.n_bus:2*self.n_bus]
        m0,res0 = self.solve(dem_day,win_day,self.sta0(24),self.gen0(24),0)
        shd += m0.tot_shd
        dem += m0.tot_dem
      return 100*shd/dem

  # Once I have learnt the status of the lines, I solve the reduce unit commitment problems for each day
  def solve_uc(self,results_file):
    n_infes = 0
    cost_prod = 0
    cost = 0
    tot_shed = 0
    tot_dem = 0
    day_shed = []
    max_hour_shed = []
    time = 0
    for i,t in enumerate(range(self.ini_test,self.end_test)):
        print('Solving day ',i)
        dem_day = self.data.iloc[24*t:24*(t+1),:self.n_bus]
        win_day = self.data.iloc[24*t:24*(t+1),self.n_bus:2*self.n_bus]
        m1,res1 = self.solve(dem_day,win_day,self.line_forecast.iloc[24*i:24*(i+1),:],self.gen0(24),0)
        if str(res1.solver.termination_condition) != 'optimal':
            n_infes += 1
        else:
            m2,res2 = self.solve(dem_day,win_day,self.sta0(24),self.gen1(m1),1)        
            cost += m2.z.value
            cost_prod += m2.cost
            tot_shed += m2.day_shd
            tot_dem += m2.day_dem
            day_shed.append(100*m2.day_shd/m2.day_dem)
            max_hour_shed.append(m2.max_hour_shd)
            time += res1['Solver'][0]['Time']
    print('Printing the results in file')
    f = open(results_file, 'a')
    f.write(self.data_file+','+str(self.slack_bus)+','+self.method+','+str(self.mat_score_trust[0][1]+self.mat_score_trust[2][1])+','+str(self.mat_score_trust[1][1])+','+str(cost)+','+str(cost_prod)+','+str(round(100*tot_shed/tot_dem,6))+','+str(round(max(day_shed),6))+','+str(round(max(max_hour_shed),6))+','+str(round(time,2))+','+str(n_infes)+'\n')
    f.close()
    results = pd.read_csv(results_file)
    print(results)
    
  def solve_uc_illustrative_example(self,results_file):
    n_infes = 0
    cost_prod = 0
    cost = 0
    tot_shed = 0
    tot_dem = 0
    day_shed = []
    max_hour_shed = []
    time = 0
    for i,t in enumerate(range(self.ini_test,self.end_test)):
        print('Solving time period ',i)
        dem_day = self.data.iloc[t:(t+1),:self.n_bus]
        win_day = self.data.iloc[t:(t+1),self.n_bus:2*self.n_bus]
        m1,res1 = self.solve(dem_day,
                             win_day,
                             self.line_forecast.iloc[i:(i+1),:],
                             self.gen0(self.end_test - self.ini_test),
                             fix = 0)
        if str(res1.solver.termination_condition) != 'optimal':
            n_infes += 1
        else:
            m2,res2 = self.solve(dem_day,
                                 win_day,
                                 self.sta0(self.end_test - self.ini_test),
                                 self.gen1(m1),
                                 fix = 1)        
            cost += m2.z.value
            cost_prod += m2.cost
            tot_shed += m2.day_shd
            tot_dem += m2.day_dem
            day_shed.append(100*m2.day_shd/m2.day_dem)
            max_hour_shed.append(m2.max_hour_shd)
            time += res1['Solver'][0]['Time']
    print('Printing the results in file')
    f = open(results_file, 'a')
    f.write("data set, method, weights,score_trust_ones, score_trust_zeros, cost, cost_prod, shedding, max_day_shed, max_hour_shed, time, infeasibilities\n")
    f.write(self.data_file+','+self.method+','+self.approach+','+str(self.mat_score_trust[0][1]+self.mat_score_trust[2][1])+','+str(self.mat_score_trust[1][1])+','+str(cost)+','+str(cost_prod)+','+str(round(100*tot_shed/tot_dem,6))+','+str(round(max(day_shed),6))+','+str(round(max(max_hour_shed),6))+','+str(round(time,2))+','+str(n_infes)+'\n')
    f.close()
    results = pd.read_csv(results_file)
    print(results)

    
    
    
    
    
    
    
    
  
