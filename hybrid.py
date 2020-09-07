import sys
import time
import numpy as np
import random as r
import networkx as nx
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import copy
import igraph as ig
import graph_tool.all as gt
import re
import operator
import traceback
import bisect

import simple_GA as sga
import experiment_graph as eg
import graph_greedy as gg
import general_sbm as g_sbm

# hybrid.py attempts to maximize some function of a set of nodes on a graph
# by first partitioning the graph. Then two independent methods determine where
# to allocate the K nodes among the partitions and where to allocate the nodes
# assigned to each partition, respectively. (See sec 4.2 of my thesis if this is unclear). 

# IN: graph - Experiment_Graph object
#     process - function(list(int),graph) -> float, a function to maximize
#     ks - list(int), sizes of solutions desired
#     outer - "ga" : uses a genetic algorithm to maximize the assignment of nodes across partition communities
#              else : uses a greedy algorithm  " " "
#     inner - "ga"  : uses a genetic algorithm to maximize the assignment of nodes within a single community
#             else  : uses a greedy algorith " " " 
#     partitioner - "sbm" : partitions the graph according to SBM or DC_SBM, whichever has a better fit
#                   else  : partitions the graph using infomap
#     **kwargs - "submod" : uses a submodular greedy algorithm for the inner maximizations
# OUT: total_solutions - list(list(int)), solutions according to sizes in ks
#      out_evals - integer, number of times process was called on the outer solutions
#      inner_evals - integer, number of times process was called on the inner solutions
def run_hybrid(graph,process,ks,outer,inner,partitioner,**kwargs):
  
  if outer == "ga":
    outer_func = outer_ga
  else:
    outer_func = outer_greedy
    
  if kwargs.get("greedy_lt",-1) > 0: # this is a quick hack to enable some experiments on LT RWN trials
    inner_func = greedy_LT_wrap(kwargs["greedy_lt"])
  elif inner == "ga":
    inner_func = ga_wrap
  elif kwargs.get("submod",False):
    inner_func = gg.greedy_sub
  else:
    inner_func = gg.greedy_general
    
  if partitioner == "sbm":
    partition_func = sbm_partitioner
  else:
    partition_func = infomap_partitioner
    
  return hybrid_max(graph,ks,process,partition_func,outer_func,inner_func,**kwargs)
    
    
  # this is special just for LT RWN trials
def greedy_LT_wrap(r_threshold):
  
  def fun(p_graph,process,in_ks):
    return p_graph.greedy_LT(in_ks,r_threshold)
  
  return fun
     
# defines partitions according to SBM recovery from graph-tool.
# both DC_SBM and classic SBMs are considered.
def sbm_partitioner(graph,pmin,pmax,runs=5):

  if "hybrid_partition" in graph.structures:
    structure = graph.structures["hybrid_partition"]
    partition = graph.partition(structure,"sbm")
  elif pmax == None and pmin == None and "sbm" in graph.structures:
    structure = graph.best_SBM()[0]
    partition = graph.partition(structure,"sbm")
    graph.structures["hybrid_partition"] = structure
  else:
    g = graph.gtgraph
    best = np.Inf
    for i in range(runs):
      structure = gt.minimize_blockmodel_dl(g, deg_corr=True, B_min=pmin, B_max=pmax)
      if structure.entropy() < best:
        partition = graph.partition(structure,"dc_sbm")
    for i in range(runs):
      structure = gt.minimize_blockmodel_dl(g, B_min=pmin, B_max=pmax)
      if structure.entropy() < best:
        partition = graph.partition(structure,"sbm")
    graph.structures["hybrid_partition"] = structure
  
  # graph.draw_graph_communities(partition)
  # plt.show()
    
  unique = np.unique(partition)
  partitions = [[] for i in range(len(unique))]
  for i in range(graph.N):
    partitions[partition[i]].append(i)
    
  graph_partitions = []
  ng = graph.nxgraph
  for part in partitions:
    subgraph = nx.convert_node_labels_to_integers(ng.subgraph(part))
    graph_partitions.append(eg.Experiment_Graph("part",nxgraph=subgraph))
  
  return graph_partitions, partitions
  
  
# defines partition according to infomap from igraph
def infomap_partitioner(graph,pmin,pmax,runs=5):

  structure = graph.get_structure("infomap",add=True)
  partition = graph.partition(structure,"infomap")
        
  # graph.draw_graph_communities(partition)
  # plt.show()
    
  unique = np.unique(partition)
  partitions = [[] for i in range(len(unique))]
  for i in range(graph.N):
    partitions[partition[i]].append(i)
    
  graph_partitions = []
  ng = graph.nxgraph
  for part in partitions:
    subgraph = nx.convert_node_labels_to_integers(ng.subgraph(part))
    graph_partitions.append(eg.Experiment_Graph("part",nxgraph=subgraph))
  
  return graph_partitions, partitions
  
# This wraps a GA for the given graph and process for neater calls throughout
# It simply takes the graph, process, and ks and returns solutions and # of evals.
def ga_wrap(graph, process, ks):
  
  def fitness(individual,sf,fit_dict):
    
    if tuple(individual) in fit_dict:
      return fit_dict[tuple(individual)], fit_dict[tuple(individual)]
      
    fitness = process(individual,graph)
    
    return fitness, fitness
  
  sols = []
  evals = 0
  
  def kcheck_soft(k):
    def kcheck_int(ind):
      if len(ind) <= k:
        return True
      return False
    return kcheck_int
  
  if ks == [0]:
    print("warning, got called for 0 length inner solution, ga_wrap.")
    return [],0
  
  for k in ks:
    if k > graph.N:
      sols.append(sols[-1])
      
    mutate = sga.klist_mutate(graph,k,mutate_type="random")
    cross = sga.k_list_crossover(graph, k, .5, crossover_type="single_point")
    kgen = sga.klist_gen(graph,k)
    kcheck = kcheck_soft(k)
    
    ps = max(25,int(np.log(graph.N/50)/np.log(10) * 100))

    GA_params = {"pr_mutate":.8,"iterations":100,"pool_size":ps,"tournament_size":2}
    
    ga = sga.GA_generator(fitness, kgen, GA_params, mutator = mutate, reproducer = cross,
      constraint_checker = kcheck, fitness_processes_per_eval=1, num_elites=2, status=False, early_stop=True)
    res = ga()
    sols.append(res[0][0])
    evals += res[1][-1]
  
  return sols, evals
  
# IN:
# graph,ks,process as from above
# inner: (graph, fitness, [ints]) -> [ [[ints]], int]
# outer: (graph, fitness, [ints]) -> [ [[ints]], [int]]
# partitioner: (graph,lo,hi) -> [graphs]  
# OUT: total_solutions - list(list(int)), solutions according to sizes in ks
#      out_evals - integer, number of times process was called on the outer solutions
#      inner_evals - integer, number of times process was called on the inner solutions
def hybrid_max(graph,ks,process,partitioner,outer,inner,**kwargs):

  ks.sort()
  in_ks = [i for i in range(1,ks[-1]+1)]
  
  pmax = kwargs.get("pmax")
  pmin = kwargs.get("pmin")
  
  partitions, node_maps = partitioner(graph,pmax,pmin)
  num_partitions = len(partitions)
  
  inner_evals = []
  set_evals = []
  out_evals = []
  
  inner_solutions = {}
  
  if kwargs.get("pre_compute_all"):
    for i in range(num_partitions):
      p_graph = partitions[i]
      inner_sol = inner(p_graph,process,in_ks)
      
      inner_evals.append(inner_sol[1]*kwargs.get("evals_per",1))
        
      for k in in_ks:
        inner_solutions[i,k] = list(inner_sol[0][k-1])
  
  def get_inner_sol(partition_index,x):
  
    if (partition_index,x) in inner_solutions:
      return inner_solutions[partition_index,x]

    p_graph = partitions[partition_index]

    if x >= p_graph.N: # Partitions may be smaller than requested solution size
      inner_sol = [i for i in range(p_graph.N)]
      inner_solutions[partition_index,x] = inner_sol
      return inner_sol
  
    inner_sol = inner(p_graph,process,[x])
    
    inner_evals.append(inner_sol[1]*kwargs.get("evals_per",1))
    inner_solutions[partition_index,x] = list(inner_sol[0][0])
    
    return list(inner_sol[0][0])
    
  def get_total_sol(outer_sol):
  
    total_sol = []
    inevals = 0
    for i in range(num_partitions):
      x = outer_sol[i]
      if x == 0:
        continue
      inner_sol = get_inner_sol(i,x)
      for member in inner_sol:
        total_sol.append(node_maps[i][member])
    
    return total_sol
    
  def outer_fitness(outer_sol):
  
    total_sol = get_total_sol(outer_sol)
    out_evals.append(kwargs.get("evals_per",1))
    
    return process(total_sol,graph)
    
  outer_solution = outer(outer_fitness,ks,num_partitions)

  total_solutions = []
  for out_sol in outer_solution:
    total_solutions.append(get_total_sol(out_sol))
  
  return total_solutions, sum(out_evals), sum(inner_evals)
    
# Custom greedy algorithm for conducting assignment among outer partitions.
def outer_greedy(outer_fitness,ks,num_partitions):

  sol = [0 for i in range(num_partitions)]
  sols = []
  
  for i in range(1,ks[-1]+1):
    max_fitness = -1

    for x in range(num_partitions):
      temp_sol = sol.copy()
      temp_sol[x] += 1
      fitness = outer_fitness(temp_sol)
      if fitness > max_fitness:
        best_sol = temp_sol.copy()
        max_fitness = fitness
    
    sol = best_sol.copy()
    if i in ks:
      sols.append(sol)
    
  return sols
  
def fitness_simple(individual):
  return sum([i*individual[i] for i in range(len(individual))]),10
      
# Custom ga for assigning K nodes to num_p partitions, maximized using outer_fitness
def outer_ga(outer_fitness, ks, num_p, pc=.5, early_stop=True, status=False,
    GA_params={"pr_mutate":.8,"iterations":100,"pool_size":100,"tournament_size":2}):
  
  def klist_gen(k):
  
    def k_ind():
      
      baby = list(np.random.choice(num_p,size=k))
      baby.sort()
      return baby
      
    return k_ind
    
  def k_list_crossover(k, pr_crossover):
    
    def single_point(individual_a, individual_b):
      ind_a = individual_a.copy()
      ind_b = individual_b.copy()
      max_xover_pt = len(ind_a)-1
      
      if ind_a == ind_b:
        return ind_a, ind_b
      
      if r.random() < pr_crossover and ind_a != ind_b:
 
        crossover_point = r.randint(0,max_xover_pt)
        temp_a = ind_a.copy()
        ind_a = ind_a[0:crossover_point] + ind_b[crossover_point:]
        ind_b = ind_b[0:crossover_point] + temp_a[crossover_point:]
   
        ind_a.sort()
        ind_b.sort()
        
      return ind_a, ind_b
    
    return single_point
    
  def klist_mutate(k):
      
    def random_mutate(ind,pr_mutation):
      individual = ind.copy()
      
      if r.random() < pr_mutation:
        individual.pop(r.randint(0,len(individual)-1))
        new_node = r.randint(0,num_p-1)
        bisect.insort(individual,new_node)

      return individual
      
    return random_mutate
      
  def fitness_wrap(outer_fit):
  
    def klist_fitness(individual,scaling,fit_dict):
        
      if tuple(individual) in fit_dict:
        return fit_dict[tuple(individual)], fit_dict[tuple(individual)]

      conv_individual = klist_to_cat_counts(individual,num_p)
      fitness = outer_fitness(conv_individual)
      
      return fitness, fitness
   
    return klist_fitness
    
  def kcheck(ind):
    if len(ind) <= k:
      return true
    return false
      
  sols = []
  
  for k in ks:
  
    fit_func = fitness_wrap(outer_fitness)
    mutate = klist_mutate(k)
    cross = k_list_crossover(k,pc)
    kgen = klist_gen(k)
    
    klist_ga = sga.GA_generator(fit_func, kgen, GA_params, mutator = mutate, reproducer = cross,
      fitness_processes_per_eval=1, num_elites=2, status=status, early_stop=early_stop)
    ksol = klist_ga()
    sols.append(klist_to_cat_counts(ksol[0][0],num_p))

  return sols

def klist_to_cat_counts(individual,num_p):
  
  unique,counts = np.unique(individual, return_counts=True)
  converted = np.zeros(num_p)
  converted[unique] = counts
  return [int(i) for i in converted]