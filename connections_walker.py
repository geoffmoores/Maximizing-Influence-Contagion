import networkx as nx
import numpy as np
import random as r
import math
import collections
import copy
import time
import sys
import matplotlib.pyplot as plt

import experiment_graph as eg
import graph_greedy as gg
#import animate as ani
import graph_gen as gen
  
# This is a probabilistic-model-building genetic algorithm (PMBGA), called connection builder 
# because I didn't know that term at the time of writing. It's not optimized for any application,
# but may be of interest for exploratory trials or extension.
# NOTE: If you would like to see / save an animation of the "interaction_utility" matrix over time,
#       uncomment "import animate as ani" above and use the animate.py script. You may have to 
#       adjust animate.py for your specific platform.

# IN: graph - Experiment_Graph
#     fun - fun(set,graph) -> float, graph function we are trying to maximize
#     ks - list(ints), list of desired solution sizes
#     params - dictionary:
#              "sol_ratio":1, the algorithm progresses by considering solutions of size sol_ratio*descent_rate^(iteration)*graph.N
#                             when sol_ratio falls below max(ks), the outter loop will terminate
#              "descent_rate":.75, how quickly the size of solutions considered diminishes
#              "coverings":1, a factor of how many solutions to consider each outer iteration
#              "update_period":20, controls how often the connection_matrix updates 
#              "min_pop":10, minimum number of solutions to consider at any given outer iteration
#    (progress) = Boolean, if true will output updates as algorithm executes
#    (animate)  = Boolean, if true will attempt to make an animation of interaction_utility over time
# OUT: results - list(list(integer)), list of solutions sized according to ks
#      evals - list[int], total number of times fun invoked
def connection_builder(graph,fun,ks,params={"sol_ratio":1,"descent_rate":.75, "coverings":1, 
  "update_period":20, "min_pop":10},progress=False,animate=False):

  def generate_soln(size):

    n1 = np.random.choice(graph.N,p=strengths)
    sol = [n1]
    
    while len(sol) < size:

      sol_inter = interaction_utility[sol]
      sol_inter[:,sol] = 0
      
      #print(sol,sol_inter,p)
      # sp = np.random.choice(len(sol),p=np.sum(sol_inter,axis=1)/np.sum(sol_inter))
     
      # next_node = np.random.choice(graph.N,p=sol_inter[sp]/np.sum(sol_inter[sp]))
      
      next_node = np.random.choice(graph.N,p=np.sum(sol_inter,axis=0)/np.sum(sol_inter))
      
      sol.append(next_node)
    
    return sol
    
  def greedy_soln(size):
    
    n1 = np.argmax(strengths)
    sol = [n1]
    
    while len(sol) < size:

      sol_inter = interaction_utility[sol]
      sol_inter[:,sol] = 0
      
      #max_location = np.unravel_index(np.argmax(sol_inter, axis=None), sol_inter.shape)
  
      #next_node = max_location[1]
      next_node = np.argmax(np.sum(sol_inter,axis=0))

      sol.append(next_node)
    
    return sol
    
  ks.sort()
  
  k = max(ks)
  
  results = []
  evals = 0
  
  sol_ratio = params["sol_ratio"]
  descent_rate = params["descent_rate"]
  coverings = params["coverings"]
  update_period = params["update_period"]
  min_pop = params["min_pop"]
  
  g_interactions = graph.N**2 - graph.N
  
  interaction_utility = np.ones((graph.N,graph.N))
  np.fill_diagonal(interaction_utility,0)
  
  if animate:
    matrices = []
    matrices.append(np.copy(interaction_utility))
  
  best = [0,[]]
  
  while sol_ratio > (k / graph.N):
    
    sol_ratio *= descent_rate
    sol_size = int(sol_ratio*graph.N)
    sol_size = max(sol_size,k)
    
    sol_interactions = sol_size**2
    pop_size = int( coverings * g_interactions / sol_interactions)
    pop_size = max(pop_size, min_pop)
    
    if progress:
      print(sol_ratio, sol_size, pop_size)
    
    strengths = np.sum(interaction_utility,axis=0)/np.sum(interaction_utility)
      
    updater = np.zeros((graph.N,graph.N))
    
    for i in range(pop_size):
    
      soln = generate_soln(sol_size)
      
      value_soln = fun(set(soln),graph)
      evals += 1
      
      if sol_size == k and value_soln > best[0]:
        best = [value_soln, soln]
      
      updater[np.ix_(soln,soln)] += value_soln
      
      if update_period > 0 and i%update_period == 0:
        
        np.fill_diagonal(updater,0)
        
        if np.sum(updater) > 0:
          updater *= g_interactions / np.sum(updater)
          interaction_utility += updater
          interaction_utility /= 2
        
        strengths = np.sum(interaction_utility,axis=0)/np.sum(interaction_utility)
        updater = np.zeros((graph.N,graph.N))

    np.fill_diagonal(updater,0)
        
    if np.sum(updater) > 0:
      updater *= g_interactions / np.sum(updater)
      interaction_utility += updater
      interaction_utility /= 2
      
    if animate:
      matrices.append(np.copy(interaction_utility))
      
    # print("\nUPDATE")
    # print(sol_size,pop_size)
    # print(updater)
    # print("INTER")
    # print(interaction_utility)
    
  if animate:
    ani.animate_matrix_evolution(matrices,show=False,save="connections",fps=1)
    
  strengths = np.sum(interaction_utility,axis=0)/np.sum(interaction_utility)
  
  for k in ks:

    greed_k = greedy_soln(k)
      
    if k == max(ks):
      val = fun(set(greed_k),graph)
      if val < best[0]:
        results.append(best[1])
        if progress:
          print(best[0],">",val)
      else:
        results.append(greed_k)
    else:
      results.append(greed_k)
    
  return results, [evals]
  
def myfun(s,graph):
  contact = set()
  for x in s:
    contact.update(set(graph.nxgraph.neighbors(x)))
  return len(contact)
  
def exp_inf_wrapper(process,process_params,trials=1):

  def exp_inf(ind,graph):
  
    processes = {"LT":graph.LT, "IC":graph.IC, "UC":graph.UC, "CC":graph.CC,
      "ADV_UC":graph.ADV_UC, "ADV_LT":graph.ADV_LT, "S_LT":graph.S_LT}
      
    g_process = processes[process]
    
    if type(ind) == list or type(ind) == set:
      individual = list(ind)
      seed_set = np.zeros(graph.N)
      seed_set[individual] = 1
    else:
      seed_set = np.copy(ind)
        
    if "innoculated" in process_params:
      process_params["innoculated"] = seed_set
      process_params["seed_set"] = graph.temp["EM_seed"]
    else:  
      if "adv_set" in process_params:
        process_params["adv_set"] = graph.temp["adv_set",process]
      process_params["seed_set"] = seed_set
    
    sum_infected = 0
    for x in range(trials):
      sum_infected += g_process(**process_params)[0]
    
    avg_inf = sum_infected / trials
    
    return avg_inf  
    
  return exp_inf
 
def temp_grab(graph,key,fun,params,overwrite=False):
  
  if key in graph.temp and not overwrite:
    return graph.temp[key]
    
  store = list(fun(*params))
  seed_set = np.zeros(graph.N)
  seed_set[store] = 1  
  #print(seed_set)
    
  graph.temp["key"] = seed_set
  return graph.temp["key"]

def deg(graph,i):
  return np.sum(graph.adj_mat[i])
  
def top_K(graph,fun,k):
  
  dict = {i:fun(graph,i) for i in range(graph.N)}
  sorted_d = sorted(dict, key=dict.get, reverse=True)
  topk = set(sorted_d[0:k])
  return topk

  
def main():
  print("Running Connection Walker as main, tests follow:")
  
  bagraph = nx.barabasi_albert_graph(100,3)
  ergraph = nx.erdos_renyi_graph(100,.03)
  dcsbm = gen.gen_dc_sbm(500,5,5,1,2,2,25,probabilities=False)
  
  graph = eg.Experiment_Graph("test",nxgraph=ergraph)
  graph = dcsbm
  
  def num_nbrs(group,graph):
    neighbors = [[i for i in graph.nxgraph.neighbors(ind)] for ind in group]
    total_unique = {nbr for nblist in neighbors for nbr in nblist}
    return len(total_unique)
    
  avg_deg = graph.M / graph.N
  c = min(.4 / avg_deg, 1.0)
  k = 7
  
  IC = exp_inf_wrapper("IC",{"seed_set":None, "c":c},trials=5)
  IC_res = exp_inf_wrapper("IC",{"seed_set":None, "c":c},trials=100)
  
  degree_seq = [graph.nxgraph.degree(node) for node in range(graph.N)]
  med_deg = np.median(degree_seq)
  t_bottom = 1/(2*med_deg)
  t_top = .5
  t1 = t_bottom + .5*(t_top - t_bottom)
  t2 = t1 * 1.5

  S_LT = exp_inf_wrapper("S_LT",{"seed_set":None, "t1":t1, "t2":t2, "type":"relative"},trials=1)
  
  LT = exp_inf_wrapper("LT",{"seed_set":None, "r_threshold":t2, "type":"relative"},trials=1)
  
  search_process = S_LT
  test_process = S_LT

  s = time.time()
  res = connection_builder(graph,search_process,[25,50],
    params={"sol_ratio":1,"descent_rate":.75, "coverings":1, "update_period":20, "min_pop":10},
    progress=False,animate=False)
  e = time.time()
  print(res)
  print("Connection Build:",test_process(res[0][-1],graph),"in",round(e-s,1))
  
    # Greedy
  s = time.time()
  res = gg.greedy(graph,search_process,[25,50],submodular=False)
  e = time.time()
  print("Greedy:",test_process(res[0][-1],graph),"in",round(e-s,1))
  
  fig = plt.figure()
  plt.imshow(graph.adj_mat.toarray(), cmap="magma")
  plt.show()
  
  # Top Deg
  top = temp_grab(graph,"top"+str(k),top_K,[graph,deg,k])
  print("Top deg:",round(test_process(top,graph),1))
      
  # Rand
  rand = r.sample(range(graph.N),k)
  print("Rand:",round(test_process(rand,graph),1))
      

if __name__ == "__main__":
  main()
