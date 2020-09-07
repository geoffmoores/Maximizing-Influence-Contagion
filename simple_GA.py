import networkx as nx
import numpy as np
import random as r
import math
import collections
import copy
import sys
import time
import heapq
from operator import itemgetter as ig
from pathos.multiprocessing import ProcessingPool as Pool
from queue import Empty
import bisect
import traceback
import operator


import experiment_graph as eg
import general_sbm as g_sbm

## simple_GA.py ## 
# Provides general functionality for modular genetic algorithms.
# Given parameters, reproduction, mutation, and parent selection functions,
# GA_generator returns a function which when called executes the defined
# genetic algorithm and returns solutions and fitness over time. 

def main():
  print("Running Simple Genetic Algorithm as main.")
  
# Default functions for mutation, reproduction, and parent selection are provided.  
# Simply so if the user doesn't offer one, it is assumed any solution is valid
def default_constraint_checker(individual):
  return True
  
# Bit String mutator
def default_mutator(individual,pr_mutation):
  mutated = ""
  for char in individual:
    if r.random() < pr_mutation:
      if char == '0':
        mutated += '1'
      else:
        mutated += '0'
    else:
      mutated += char
  return mutated
 
# String Recombination from two parents, yields two children
def default_reproducer(parent_a,parent_b):
  split = r.randint(0,len(parent_a))
  child_a = parent_a[:split] + parent_b[split:]
  child_b = parent_b[:split] + parent_a[split:]
  return child_a, child_b
  
# Default Parent Selection is tournament 
def default_parent_selector(pool,tournament_size):
  parent_a = max(r.sample(pool,tournament_size),key=lambda item:item[1])
  parent_b = max(r.sample(pool,tournament_size),key=lambda item:item[1])
  return parent_a, parent_b
 
# Required Inputs:
#   1) fitness evaluator: function(individual,[0,1],dictionary) -> any ">" ordered set
#   2) individual generator: function() -> individual
# Optional Inputs:
#   a) ga_params: dictionary as below
#   b) mutator: function(individual,pr_mutation) -> individual
#   c) reproducer: function(individual,individual) -> (individual, individual)
#   d) parent_selector: function(pool,integer) -> (individual, individual)
#   e) constraint_checker: function(individual) -> boolean
#   f) fitness_processes_per_eval: integer, enables mapping fitness over total processes run
# Output:
#   genetic_algorithm: function() -> individual
#     genetic_algorithm returns:
#       best_individual - best solution found by the GA
#       total_process_evals - integer, number of times fitness function was called 
#       fitness_over_time - list(float), best / elite fitness over GA generations 
#       avg_fitness_over_time - list(float), average fitness over GA generations 
#       pool - list([individual,float]), all individuals and their fitness from final generation
def GA_generator(fitness_evaluator, individual_generator,
  ga_params = {"pr_mutate":.01,"iterations":100,"pool_size":100,"tournament_size":3},
  mutator=default_mutator, reproducer=default_reproducer, parent_selector=default_parent_selector,
  constraint_checker=default_constraint_checker,fitness_processes_per_eval=1,num_elites=2,multiprocess=False,
  debug=False,status=True,early_stop=True):
  
  
  def genetic_algorithm():
  
    fitness_dict = collections.OrderedDict()
    fit_cache_size = ga_params["pool_size"] * 10 # Can increase as desired, shouldnt be too large unless you have
    #   reason to suspect that certain patterns will repeat themselves
    fitness_over_time = []
    avg_fitness_over_time = []
    
    total_process_evals = [ga_params["pool_size"]*fitness_processes_per_eval]
    pool = [individual_generator() for e in range(ga_params["pool_size"])]
    pool = [[a,fitness_evaluator(a,0,fitness_dict)] for a in pool]
    for individual in pool:
      fitness_dict[tuple(individual[0])] = individual[1][1]
    
    pool = [[p[0],p[1][0]] for p in pool]
    
    if multiprocess:
      processors = 4 #mp.cpu_count()
      process_pool = Pool(processors)
    
    end_time = 0
    mutate_cross_time = 0
    fit_time = 0
    elite_time = 0
    cleanup_time = 0
    t1 = 0
    
    start = time.time()
    no_change_best = 0
    
    for iteration in range(ga_params["iterations"]):

      if debug:
        print("Generation:",iteration,"elites:",round(elite_time,3),"mutate+cross:",round(mutate_cross_time,3),
          "fit:",round(fit_time,3),"cleanup:",round(cleanup_time,3),"total:",round(end_time-t1,1))
        print("Pool:",len(pool),"FD",len(fitness_dict))
        for ind in pool[0:5]:
          print(ind)      
        mutate_cross_time = 0
        fit_time = 0
        elite_time = 0
        cleanup_time = 0
        end_time = 0
    

      ## Elites
      t1 = time.time()
      if iteration > 0:
        last_elite = next_generation[0]
      next_generation = heapq.nlargest(num_elites, pool, key=lambda item:item[1])
      this_elite = next_generation[0]
      
      if iteration > 0 and last_elite[0] == this_elite[0]:
        no_change_best += 1
      else:
        no_change_best = 0
      
      next_generation = [[a[0],fitness_evaluator(a[0],(iteration+1)/ga_params["iterations"],fitness_dict)] for a in next_generation]
      for individual in next_generation:
        if len(fitness_dict) > fit_cache_size:
          fitness_dict.popitem(last=False) # FIFO by ordered dict
        fitness_dict[tuple(individual[0])] = individual[1][1] # store unadjusted fitness
      next_generation = [[e[0],e[1][0]] for e in next_generation] # set to just hold adjusted fitness
      t2 = time.time()
      elite_time = t2-t1
      
      
      ## Statistic Tracking
      if len(total_process_evals) > 0:
        total_process_evals.append(total_process_evals[-1] + ga_params["pool_size"]*fitness_processes_per_eval)
      fitness_over_time.append(next_generation[0][1])
      avg_fitness_over_time.append(np.average([individual[1] for individual in pool]))
      
      for spawn_pool in range(0, ga_params["pool_size"], 4):
      
        ta = time.time()

        parent_a, parent_b = parent_selector(pool,ga_params["tournament_size"])
        child_a, child_b = reproducer(parent_a[0],parent_b[0])
        child_a = [mutator(child_a,ga_params["pr_mutate"]),-1]
        child_b = [mutator(child_b,ga_params["pr_mutate"]),-1]
        family_1 = [parent_a,parent_b,child_a,child_b]
        parents = [parent_a, parent_b]
        children = [child_a, child_b]
        
        if spawn_pool+3 < ga_params["pool_size"]:
          parent_c, parent_d = parent_selector(pool,ga_params["tournament_size"])
          child_c, child_d = reproducer(parent_c[0],parent_d[0])
          child_c = [mutator(child_c,ga_params["pr_mutate"]),-1]
          child_d = [mutator(child_d,ga_params["pr_mutate"]),-1]
          family_2 = [parent_a,parent_b,child_a,child_b]
          
          parents += [parent_c, parent_d]
          children += [child_c, child_d]
        tb = time.time()
        
        mutate_cross_time += tb - ta
        
        ## FITNESS EVALUATION ##
        ## We can bring multiprocessing back in if needed, but it turns out to be 
        ## super expensive to set up the other processes.  Only useful on very large
        ## graphs with many fitness evaluations.
        # Temporarily, individuals will be stored as [individual, [adjusted fitness, unadjusted fitness]]
        #try:
        # if False:
          # tc = time.time()
          
          # fit_results = [process_pool.apipe( fitness_evaluator, member[0], (iteration+1)/ga_params["iterations"], fitness_dict ) for member in parents]
          # parent_fitnesses = [fit_result.get() for fit_result in fit_results]
          # parents = [ [parents[i][0], parent_fitnesses[i]] for i in range(len(parents))]
          # td = time.time()
          
          # fit_results = [process_pool.apipe( fitness_evaluator, member[0], (iteration+1)/ga_params["iterations"], fitness_dict ) for member in children]
          # children_fitnesses = [fit_result.get() for fit_result in fit_results]
          # children = [ [children[i][0], children_fitnesses[i]] for i in range(len(children))]
          # te = time.time()
          
          # family_1 = parents[:2] + children[:2]
          # if spawn_pool+3 < ga_params["pool_size"]:
            # family_2 = parents[2:] + children[2:]
          
          # parent_fit_time += td - tc
          # child_fit_time += te-td
          
        # #except Exception as e:
        # #  print("Issue with multiprocessing.")
        # #  print(e)
        #else:
        fts = time.time()
        family_1 = [[member[0], fitness_evaluator(member[0],(iteration+1)/ga_params["iterations"],fitness_dict)] for member in family_1]
        if spawn_pool+3 < ga_params["pool_size"]:
          family_2 = [[member[0], fitness_evaluator(member[0],(iteration+1)/ga_params["iterations"],fitness_dict)] for member in family_2]
        fte = time.time()
        fit_time += fte - fts
        
        ## BUILD NEXT GEN AND RESET INDIVIDUAL STRUCTURE
        # after the nested for loop, individuals will be reset to [individual, adjusted fitness] 
        
        t3 = time.time()
        families = [family_1]
        if spawn_pool+3 < ga_params["pool_size"]:
          families += [family_2]

        for family in families:
          
          for individual in family:
            if len(fitness_dict) > fit_cache_size:
              fitness_dict.popitem(last=False) # FIFO by ordered dict
            fitness_dict[tuple(individual[0])] = individual[1][1] # store unadjusted fitness
          
          family = [[f[0],f[1][0]] for f in family] # set to just hold adjusted fitness

          best = max(family,key=lambda item:item[1])
          family.remove(best)
          next_generation.append(best)
          
          if len(next_generation) < ga_params["pool_size"]:
            next = r.sample(family,1)[0]
            next_generation.append(next)
          
        t4 = time.time()
        cleanup_time += t4-t3
        
      while len(next_generation) > ga_params["pool_size"]:
        next_generation.pop()
      
      pool = next_generation.copy()
      end_time = time.time()
      
      t = time.time()
      if iteration%25 == 0 and status:
        print(iteration,"in",round(t-start,2))
        
      if no_change_best == 20 and early_stop:
        if status:
          print("Early stopped at generation:",iteration)
        break
      
    valid_pool = [individual for individual in pool if constraint_checker(individual[0])]
    if len(valid_pool) == 0:
      print("Warning: solution did not converge on valid individual")
      best_individual = [[],0]
    else:
      best_individual = max(valid_pool, key=lambda item:item[1])
    
    ## Statistic Tracking
    total_process_evals.append(total_process_evals[-1] + ga_params["pool_size"]*fitness_processes_per_eval)
    fitness_over_time.append(best_individual[1])
    avg_fitness_over_time.append(np.average([individual[1] for individual in pool]))

    return best_individual, total_process_evals, fitness_over_time, avg_fitness_over_time, pool
  
  return genetic_algorithm
  
## FIT PROPORTIONATE SELECTION ##

# IN: pool - list([individual,fitness]), genetic algorithms pool
#     (size) - integer > 0, number of parents to select
def fit_prop_select(pool,size=2):
  fitnesses = np.array([e[1] for e in pool])
  selection_chance = fitnesses / np.sum(fitnesses)
  selections = np.random.choice(np.arange(len(pool)),size=size,p=selection_chance)
  
  return [pool[i] for i in selections]
  
## FITNESS ##

# IN: graph - Experiment_Graph object
#     trials - integer # of trials to evaluate fitness
#     process - a function of graph from (individual) -> float
#     process_params - a dictionary of parameters for process
#     k - size of individuals in # of nodes
# OUT: klist_fitness(individual,scaling_factor,fit_dict)
#         IN: individual - length k solution
#             scaling_factor - not used here, vestigial to maintain function call signatures
#             fit_dict - dictionary[individual] -> float, cache of known fitnesses to spare calculation
#         OUT: fitness of individual, float
def klist_fit_wrapper(graph,trials,process,process_params,k):

  if "innoculated" in process_params:
    process_params["seed_set"] = graph.temp["EM_seed"]
  if "adv_set" in process_params:
    if "c" in process_params:
      process_params["adv_set"] = graph.temp["adv_set","ADV_UC"]
    elif "r_threshold" in process_params:
      process_params["adv_set"] = graph.temp["adv_set","ADV_LT"]
      
  # two fitnesses returned to match signature / handling in simple_ga
  # facilitates scaling over time (not necessary here)
  def klist_fitness(individual,scaling_factor,fit_dict):
    
    if tuple(individual) in fit_dict:
      return fit_dict[tuple(individual)], fit_dict[tuple(individual)]

    sum_infected = 0
    seed_set = np.zeros(graph.N)
    seed_set[individual] = 1

    if "innoculated" in process_params:
      process_params["innoculated"] = seed_set
    else:
      process_params["seed_set"] = seed_set
    
    if trials == 1:
      fitness = process(**process_params)[0]
    else:
      for x in range(trials):
        sum_infected += process(**process_params)[0]
      
      fitness = sum_infected / trials
    
    return fitness, fitness
 
  return klist_fitness
      
# IN: graph - Experiment_Graph object
#     trials - integer # of trials to evaluate fitness
#     process - a function of graph from (individual) -> float
#     process_params - a dictionary of parameters for process
#     k - size of individuals in # of nodes
# OUT: klist_fitness(individual,scaling_factor,fit_dict)
#         IN: individual - length k solution
#             scaling_factor - float in [0,1], closeness to end of GA, allows for temporal selection pressure
#                              on oversize solutions (eventually nlist should develop solutions of length k)
#             fit_dict - dictionary[individual] -> float, cache of known fitnesses to spare calculation
#         OUT: fitness of individual, float
def nlist_fit_wrapper(graph,trials,process,process_params,k):

  # two fitnesses returned, one scaled, one pure
  # facilitates quick updating 
  def nlist_fitness(individual,scaling_factor,fit_dict):
    if len(individual) == 0:
      return 0,0
  
    scaling_factor_2 = 1-scaling_factor # 0 -> end generation
    
    if tuple(individual) in fit_dict:
      excess = max(0,len(individual)-k)
      if scaling_factor_2 == 0:
        if excess == 0:
          return fit_dict[tuple(individual)], fit_dict[tuple(individual)]
        return 0,0

      
      scaled_excess = excess/(scaling_factor_2**(excess)*k)
      penalty = 1/(1+scaled_excess) #2/(1+np.exp(2*scaled_excess))
      fitness = fit_dict[tuple(individual)]
      #fitness /= (1+excess)
      
      return fitness*penalty, fitness #(scaling_factor)*penalty+fitness*(1-scaling_factor), fitness
    
    sum_infected = 0
    seed_set = np.zeros(graph.N)
    seed_set[individual] = 1
    process_params["seed_set"] = seed_set

    for x in range(trials):
      sum_infected += process(**process_params)[0]
    
    fitness = sum_infected / trials
    
    excess = max(0,len(individual)-k)
    if scaling_factor_2 == 0:
      if excess == 0:
        return fitness, fitness
      return 0,0
    
    excess = max(0,len(individual)-k)
    #fitness /= (1+excess)
    scaled_excess = excess/(scaling_factor_2**(excess)*k)
    penalty = 1/(1+scaled_excess) #2/(1+np.exp(2*scaled_excess)) 
    
    return fitness*penalty, fitness #(scaling_factor)*penalty+fitness*(1-scaling_factor), fitness
 
  return nlist_fitness
  
  
## MUTATION ##

# IN: graph - Experiment_Graph object
#     k - size of individuals in # of nodes
#     (mutate_type) - "random" : default, mutates n-list at random locations
#                   "neighbor" : mutates neighbors of nodes in current individual's nodes
#                   "distance" : mutates nodes distant from current individual's nodes
#     (node_selection_probs) - 1D np.array, normalized, allows for deliberate probability distribution
#                   that any node is mutated into the solution.
#     (degree_wtd) - Boolean, if True sets node_selection_probs proportional to node degrees
# OUT: mutate(individual,pr_mutation)
#         IN: individual - length k solution
#             pr_mutation - float in [0,1], likelihood mutation occurs at all
#         OUT: mutated individual, according to pr_mutation and parameters to nlist_mutate
def nlist_mutate(graph,k,mutate_type="random",node_selection_probs=np.array([]),degree_wtd=False):
  
  def aggressive_mutate(ind_x):
    
    num_pop = r.randint(0,max(1,len(ind_x)-k))
    index = r.randint(0,len(ind_x)-1)
    if index + num_pop > len(ind_x):
      leftover = len(ind_x) - (index+num_pop)
      return ind_x[leftover:index]
    return ind_x[:index] + ind_x[index+num_pop:]
  
  def random_mutate(ind,pr_mutation):
    individual = ind.copy()
    if len(individual) == 0 or len(individual) == graph.N:
      return individual
    if r.random() < pr_mutation:
      if len(individual) < k:
        dec_prob,inc_prob = [.4,.8]
      else:
        dec_prob,inc_prob = [.8,.4]      
      if r.random() < dec_prob:
        individual = aggressive_mutate(individual)
      if r.random() < inc_prob:
        node_probs = np.copy(node_selection_probs)

        node_probs[individual] = 0.0 # take our individual's current member nodes out of the hat
          
        node_probs /= np.sum(node_probs) # renormalize probabilities
        #print("Random\n",node_probs,"\n")
        new_member = np.random.choice(graph.N,p=node_probs)
 
        bisect.insort(individual,new_member)
    
    return individual
    
  def neighbor_mutate(ind,pr_mutation):
    individual = ind.copy()
    if len(individual) == 0 or len(individual) == graph.N:
      return individual
    if r.random() < pr_mutation:
      if len(individual) < k:
        dec_prob,inc_prob = [.4,.8]
      else:
        dec_prob,inc_prob = [.8,.4]      
      if r.random() < dec_prob:
        individual = aggressive_mutate(individual)
      if r.random() < inc_prob:
        node_probs = np.copy(node_selection_probs)
  
        np_individual = np.zeros(graph.N)
        np_individual[individual] = 1.0
        node_probs = np.where(graph.adj_mat * np_individual != 0, node_probs, 0)
        node_probs[individual] = 0.0
          
        if np.sum(node_probs) > 0:
          node_probs /= np.sum(node_probs)
          #print("Neighbor\n",individual,node_probs,"\n")
          new_member = np.random.choice(graph.N,p=node_probs)
          bisect.insort(individual,new_member)
    
    return individual
  
  def distance_mutate(ind,pr_mutation):
    individual = ind.copy()
    if len(individual) == 0 or len(individual) == graph.N:
      return individual
    if r.random() < pr_mutation:
      if len(individual) < k:
        dec_prob,inc_prob = [.4,.8]
      else:
        dec_prob,inc_prob = [.8,.4]      
      if r.random() < dec_prob:
        individual = aggressive_mutate(individual)
      if r.random() < inc_prob and len(individual) < graph.N and len(individual) > 0:
        origin = individual[r.randint(0,len(individual)-1)]
        distances = graph.get_distances()[origin]
        median = np.median(distances)
        distant_nodes = np.argwhere(distances >= median).T[0]
        
        node_probs = np.copy(node_selection_probs)
        node_probs[individual] = 0
        node_probs = node_probs[distant_nodes]
        
        if np.sum(node_probs) > 0:
          node_probs /= np.sum(node_probs)
          #print("Distance\n",individual, origin, half, distances,node_probs,"\n")
          new_node = np.random.choice(distant_nodes,p=node_probs)
          bisect.insort(individual,new_node)
        
    return individual 
    
  def null_mutate(ind,pr_mutation):
    individual = ind.copy()
    return individual
    
  if not np.any(node_selection_probs):
    node_selection_probs = np.ones(graph.N)
    
  if degree_wtd:
    node_selection_probs = graph.get_deg_seq(relative=True)
  
  if mutate_type == "random":
    return random_mutate
  elif mutate_type == "neighbor":
    return neighbor_mutate
  elif mutate_type == "distance":
    return distance_mutate
  else:
    print("Warning: type of mutation not recognized, no mutation will occur.")
    return null_mutate
    
# IN: graph - Experiment_Graph object
#     k - size of individuals in # of nodes
#     (mutate_type) - "random" : default, mutates n-list at random locations
#                   "neighbor" : mutates neighbors of nodes in current individual's nodes
#                   "distance" : mutates nodes distant from current individual's nodes
#     (node_selection_probs) - 1D np.array, normalized, allows for deliberate probability distribution
#                   that any node is mutated into the solution.
#     (degree_wtd) - Boolean, if True sets node_selection_probs proportional to node degrees
# OUT: mutate(individual,pr_mutation)
#         IN: individual - length k solution
#             pr_mutation - float in [0,1], likelihood mutation occurs at all
#         OUT: mutated individual, according to pr_mutation and parameters to nlist_mutate
def klist_mutate(graph,k,mutate_type="random",node_selection_probs=np.array([]),degree_wtd=False):
    
  def random_mutate(ind,pr_mutation):
    individual = ind.copy()
    
    if r.random() < pr_mutation:
      x = individual.pop(r.randint(0,len(individual)-1))
      new_node = ind[0]
      while new_node in ind:
        new_node = r.randint(0,graph.N-1)
      bisect.insort(individual,new_node)

      return individual

      if degree_wtd:
        node_probs = np.copy(node_selection_probs)
        node_probs[individual] = 0
        node_probs /= np.sum(node_probs)
      
        new_node = np.random.choice(graph.N,p=node_probs)
        print("?")
      else:
        new_node = ind[0]
        while new_node in ind:
          new_node = r.randint(0,graph.N-1)
      individual.append(new_node)
      
      #bisect.insort(individual,new_node)
  
      #individual.sort()
    return individual
    
  def neighbor_mutate(ind,pr_mutation):
    individual = ind.copy()
    
    if r.random() < pr_mutation:
      individual.pop(r.randint(0,len(individual)-1))
     
      np_individual = np.zeros(graph.N)
      np_individual[individual] = 1.0
      node_probs = np.where(graph.adj_mat * np_individual != 0, node_selection_probs, 0)
      node_probs[individual] = 0.0
        
      node_probs /= np.sum(node_probs)
      
      new_node = np.random.choice(graph.N,p=node_probs)
      bisect.insort(individual,new_node)

    return individual
    
  def distance_mutate(ind,pr_mutation):
    individual = ind.copy()
    
    if r.random() < pr_mutation:
      origin = individual.pop(r.randint(0,len(individual)-1))
      distances = graph.get_distances()[origin]
      median = np.median(distances)
      distant_nodes = np.argwhere(distances >= median).T[0]
     
      node_probs = np.copy(node_selection_probs)
      node_probs[individual] = 0
      node_probs = node_probs[distant_nodes]
      
      node_probs /= np.sum(node_probs)
      
      new_node = np.random.choice(distant_nodes,p=node_probs)
      bisect.insort(individual,new_node)

    return individual
      
  def null_mutate(ind,pr_mutation):
    individual = ind.copy()
    return individual
    
  if not np.any(node_selection_probs):
    node_selection_probs = np.ones(graph.N)
    
  if degree_wtd:
    node_selection_probs = graph.get_deg_seq(relative=True)
  
  if mutate_type == "random":
    return random_mutate
  elif mutate_type == "neighbor":
    return neighbor_mutate
  elif mutate_type == "distance":
    return distance_mutate
  else:
    print("Warning: type of mutation not recognized, no mutation will occur.")
    return null_mutate
    

## CROSSOVER ##
# IN: graph - Experiment_Graph object
#     k - size of individuals in # of nodes
#     pr_crossover - float [0,1], likelihood crossover event is conducted when crossover is called
#     (crossover_type) - "single_point"  : default, single point crossover 
#                   "degree_cooperation" : orders the individuals according to degree of nodes
#                   "network_mask"       : uses a breadth-first-search from individuals' nodes, on graph,  
#                                          to identify what nodes to crossover
#     (node_selection_probs) - 1D np.array, normalized, allows for deliberate probability distribution
#                   that any node is mutated into the solution.
#     (degree_wtd) - Boolean, if True sets node_selection_probs proportional to node degrees
# OUT: crossover(individual_A,individual_B)
#         IN: individual_A, individual_B - length k solutions
#         OUT: two new individuals according crossover between individuals A and B
def k_list_crossover(graph, k, pr_crossover, crossover_type="single_point",node_selection_probs=np.array([]), degree_wtd=False):
  
  def single_point(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b
    
    if r.random() < pr_crossover and ind_a != ind_b:
      
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]
      if len(unique_a) != len(unique_b):
        print("Unexpected Input: Individuals of unequal size or with duplicate members.")
      crossover_point = r.randint(0,len(unique_a))
      
      each = [a for a in ind_a if a in ind_b]
        
      temp_a = unique_a.copy()
      add_a = unique_a[0:crossover_point] + unique_b[crossover_point:]
      add_b = unique_b[0:crossover_point] + unique_a[crossover_point:]
      
      ind_a = each + add_a
      ind_b = each + add_b
      
      if len(ind_a) != k or len(ind_b) != k:
        print("Unexpected Size: Input or result was not length",k,":\n",ind_a,"\n",ind_b)
        ind_a = ind_a[0:k]
        ind_b = ind_b[0:k]
 
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def network_mask(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b
    
    if r.random() < pr_crossover and ind_a != ind_b:
    
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]
      if len(unique_a) != len(unique_b):
        print("Unexpected Input: Individuals of unequal size or with duplicate members.")
        print(ind_a,"\n",ind_b)
      crossover_point = r.randint(0,len(unique_a))
      
      each = [a for a in ind_a if a in ind_b]
      
      uniques = unique_a + unique_b
      #print(uniques)
     
      # Get a list of size |crossover_point| by starting from an element in unique_b
      # and retrieving |crossover_point| nodes that are closest to that origin in uniques
      # we will add these to individual A
      origin_a = r.sample(unique_b,1)[0] 
      distance_og_a = graph.get_distances()[origin_a]
      distance_pairs = [(u,distance_og_a[u]) for u in uniques]
      distance_pairs.sort(key = operator.itemgetter(1))
      
      add_a = [e[0] for e in distance_pairs[0:crossover_point]]
      
      uniques = [u for u in uniques if u not in add_a]
      
      # Vice versa for elements to add to individual B
      origin_b = r.sample(uniques,1)[0]
      distance_og_b = graph.get_distances()[origin_b]
      distance_pairs = [(u,distance_og_b[u]) for u in uniques]
      distance_pairs.sort(key = operator.itemgetter(1))
      
      add_b = [e[0] for e in distance_pairs[0:crossover_point]]
      
      uniques = [u for u in uniques if u not in add_b]
      
      # Remaining elements are agnostically divided back to A and B
      # if the crossover was evenly distributed, A and B will receive back
      # their original unswapped nodes 
      if len(uniques) > 0:
        #print(add_a,add_b,uniques)
        cross_2 = int(len(uniques)/2)
        add_a += uniques[0:cross_2]
        add_b += uniques[cross_2:]
      
      ind_a = each + add_a
      ind_b = each + add_b
      
      if len(ind_a) != k or len(ind_b) != k:
        print("Unexpected Size: Input or result was not length",k,":\n",ind_a,"\n",ind_b)
        ind_a = ind_a[0:k]
        ind_b = ind_b[0:k]
 
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def degree_cooperation(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b
    
    if r.random() < pr_crossover and ind_a != ind_b:
      
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]
      if len(unique_a) != len(unique_b):
        print("Unexpected Input: Individuals of unequal size or with duplicate members.")
      crossover_point = r.randint(0,len(unique_a))
      
      each = [a for a in ind_a if a in ind_b]
      
      node_wts = np.copy(node_selection_probs)
      if np.all(node_wts == np.ones(graph.N)):
        node_wts = graph.get_deg_seq()
        
      a_by_wts = [(a,node_wts[a]) for a in unique_a]
      a_by_wts.sort(key = operator.itemgetter(1))
      
      b_by_wts = [(b,node_wts[b]) for b in unique_b]
      b_by_wts.sort(key = operator.itemgetter(1))
      
      add_a = a_by_wts[0:crossover_point] + b_by_wts[crossover_point:]
      add_b = b_by_wts[0:crossover_point] + a_by_wts[crossover_point:]
      
      add_a = [a[0] for a in add_a]
      add_b = [b[0] for b in add_b]
      
      ind_a = each + add_a
      ind_b = each + add_b
      
      if len(ind_a) != k or len(ind_b) != k:
        print("Unexpected Size: Input or result was not length",k,":\n",ind_a,"\n",ind_b)
        ind_a = ind_a[0:k]
        ind_b = ind_b[0:k]
 
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def null_crossover(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    return ind_a, ind_b
    
  if not np.any(node_selection_probs):
    node_selection_probs = np.ones(graph.N)
    
  if degree_wtd:
    node_selection_probs = graph.get_deg_seq(relative=True)
  
  if crossover_type == "single_point":
    return single_point
  elif crossover_type == "degree_cooperation":
    return degree_cooperation
  elif crossover_type == "network_mask":
    return network_mask
  else:
    print("Warning: type of crossover not recognized, no crossover will occur.")
    return null_crossover
    
# IN: graph - Experiment_Graph object
#     k - size of individuals in # of nodes
#     pr_crossover - float [0,1], likelihood crossover event is conducted when crossover is called
#     (crossover_type) - "double_point"  : default, double point crossover 
#                   "degree_cooperation" : orders the individuals according to degree of nodes
#                   "network_mask"       : uses a breadth-first-search from individuals' nodes, on graph,  
#                                          to identify what nodes to crossover
#     (node_selection_probs) - 1D np.array, normalized, allows for deliberate probability distribution
#                   that any node is mutated into the solution.
#     (degree_wtd) - Boolean, if True sets node_selection_probs proportional to node degrees
# OUT: crossover(individual_A,individual_B)
#         IN: individual_A, individual_B - length k solutions
#         OUT: two new individuals according crossover between individuals A and B
def n_list_crossover(graph, k, pr_crossover, crossover_type="double_point",node_selection_probs=np.array([]), degree_wtd=False):
  
  def double_point(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b
    
    if r.random() < pr_crossover and ind_a != ind_b:
      
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]

      crossover_a = r.randint(0,len(unique_a))
      crossover_b = r.randint(0,len(unique_b))
      
      each = [a for a in ind_a if a in ind_b]
        
      add_a = unique_a[crossover_a:] + unique_b[:crossover_b]
      add_b = unique_a[:crossover_a] + unique_b[crossover_b:]
      
      ind_a = each + add_a
      ind_b = each + add_b
      
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def network_mask(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b    
    
    if r.random() < pr_crossover and ind_a != ind_b:
    
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]

      each = [a for a in ind_a if a in ind_b]
      
      uniques = unique_a + unique_b
      if len(uniques) == 0:
        print(ind_a, ind_b)
      crossover_point = r.randint(0,len(uniques))
     
      # Get a list of size |crossover_point| by starting from an element in unique_b
      # and retrieving |crossover_point| nodes that are closest to that origin in uniques
      # we will add these to individual A
      if len(unique_b) > 0:
        origin_a = r.sample(unique_b,1)[0] 
      else:
        origin_a = r.sample(uniques,1)[0] 
        
      distance_og_a = graph.get_distances()[origin_a]
      distance_pairs = [(u,distance_og_a[u]) for u in uniques]
      distance_pairs.sort(key = operator.itemgetter(1))
      
      add_a = [e[0] for e in distance_pairs[0:crossover_point]]
      
      add_b = [u for u in uniques if u not in add_a]
      
      ind_a = each + add_a
      ind_b = each + add_b
 
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def degree_cooperation(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    
    if ind_a == ind_b:
      return ind_a, ind_b
    
    if r.random() < pr_crossover and ind_a != ind_b:
      
      unique_a = [a for a in ind_a if a not in ind_b]
      unique_b = [b for b in ind_b if b not in ind_a]

      crossover_a = r.randint(0,len(unique_a))
      crossover_b = r.randint(0,len(unique_b))
      
      each = [a for a in ind_a if a in ind_b]
      
      node_wts = np.copy(node_selection_probs)
      if np.all(node_wts == np.ones(graph.N)):
        node_wts = graph.get_deg_seq()
        
      a_by_wts = [(a,node_wts[a]) for a in unique_a]
      a_by_wts.sort(key = operator.itemgetter(1))
      
      b_by_wts = [(b,node_wts[b]) for b in unique_b]
      b_by_wts.sort(key = operator.itemgetter(1))
      
      add_a = a_by_wts[0:crossover_a] + b_by_wts[crossover_b:]
      add_b = b_by_wts[0:crossover_b] + a_by_wts[crossover_a:]
      
      add_a = [a[0] for a in add_a]
      add_b = [b[0] for b in add_b]
      
      ind_a = each + add_a
      ind_b = each + add_b
 
      ind_a.sort()
      ind_b.sort()
      
    return ind_a, ind_b
    
  def null_crossover(individual_a, individual_b):
    ind_a = individual_a.copy()
    ind_b = individual_b.copy()
    return ind_a, ind_b
    
  if not np.any(node_selection_probs):
    node_selection_probs = np.ones(graph.N)
    
  if degree_wtd:
    node_selection_probs = graph.get_deg_seq(relative=True)
  
  if crossover_type == "double_point":
    return double_point
  elif crossover_type == "degree_cooperation":
    return degree_cooperation
  elif crossover_type == "network_mask":
    return network_mask
  else:
    print("Warning: type of crossover not recognized, no crossover will occur.")
    return null_crossover
    
# Random solution generator for k list
def klist_gen(graph, k):
  
  q = min(graph.N,k)
  
  def k_ind():
    baby = r.sample(range(graph.N-1),q)
    baby.sort()
    return baby
   
  return k_ind
  
# Random solution generator for n list
def nlist_gen(graph, k):
  
  def n_ind():
    baby = r.sample(range(graph.N-1),r.randint(1,graph.N-1))
    baby.sort()
    return baby
  
  return n_ind
  
def check_k(k):
  
  def k_check(ind):
    return len(ind) == k
    
  return k_check
  
def read_input(filename):

  filename = "graphs/" + filename
  with open(filename,'r') as tsvin:
    row_iter = csv.reader(tsvin, delimiter='\n')
    graph = nx.Graph()

    for element in row_iter:
      row = element[0].split()
      u = int(row[0])
      v = int(row[1])
      graph.add_edge(u,v)
      
  return graph  
  
def get_lcc(graph):
  
  large_comp = max(nx.connected_components(graph), key=len)
  return graph.subgraph(list(large_comp))
    
def get_fb_2():

  g = get_lcc(read_input('fb_2.txt'))
  read_graph = eg.Experiment_Graph("fb_2_LCC",nxgraph = g)
  return read_graph
  
def ba_test_graph(N,m):
  g = nx.barabasi_albert_graph(N,m)
  return eg.Experiment_Graph("BA_test",nxgraph = g)
  
def run_aggregate_tests():

  g = nx.Graph()
  g.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(6,8),(6,9)])
  test_graph = eg.Experiment_Graph("test",nxgraph=g)
 
  LT_params = {"seed_set":None, "a_threshold":2, "type":"absolute"}
  
  IC_params = {"seed_set":None, "c":.5}
  
  GA_params = {"pr_mutate":.5,"iterations":10,"pool_size":8,"tournament_size":2}

  k = 2
  klist_fit_func = klist_fit_wrapper(test_graph,10,test_graph.LT,LT_params,k)
  distant_mutate = klist_mutate(test_graph,k,mutate_type="distance",degree_wtd=True)
  kdegree = k_list_crossover(test_graph, k, 1.0, crossover_type="degree_cooperation", degree_wtd=True)
  kgen = klist_gen(test_graph,k)
  kcheck = check_k(k)
  
  klist_ga = GA_generator(klist_fit_func, kgen, GA_params, mutator = distant_mutate, reproducer = kdegree,
    constraint_checker = kcheck,fitness_processes_per_eval=5,num_elites=2,debug=True)
  print(klist_ga())
  
  GA_params = {"pr_mutate":1.0,"iterations":50,"pool_size":50,"tournament_size":2}
  
  k = 2
  nlist_fit_func = nlist_fit_wrapper(test_graph,1,test_graph.LT,LT_params,k)
  distant_mutate = nlist_mutate(test_graph,k,mutate_type="distance",degree_wtd=True)
  nmask = n_list_crossover(test_graph,k, 1.0, crossover_type="network_mask", degree_wtd=True)  
  ngen = nlist_gen(test_graph,k)
  
  nlist_ga = GA_generator(nlist_fit_func, ngen, GA_params, mutator = distant_mutate, reproducer = nmask,
    fitness_processes_per_eval=1,num_elites=1,debug=True)
  print(nlist_ga())
  
  print("Test K-List Graph GA Wrapper:")
  kga = K_graph_GA(test_graph,test_graph.LT,LT_params,1,5,"random","single_point",.3)
  print(kga())
  
# compact generator for standard GA's on networks
# minimal example call: 
#      simple_GA.K_graph_GA(my_graph,my_graph.LT,{"seed_set":None, "a_threshold":2, "type":"absolute"})
def K_graph_GA(graph,process,process_params,trials=1,k=None,mutator="random",crossover="single_point",pc=.3,deg_wtd=False, early_stop=True,
    GA_params={"pr_mutate":.8,"iterations":100,"pool_size":100,"tournament_size":2},status=False):
  
  klist_fit_func = klist_fit_wrapper(graph,trials,process,process_params,k)
  mutate = klist_mutate(graph,k,mutate_type=mutator,degree_wtd=deg_wtd)
  cross = k_list_crossover(graph, k, pc, crossover_type=crossover, degree_wtd=deg_wtd)
  kgen = klist_gen(graph,k)
  kcheck = check_k(k)
  
  if GA_params["tournament_size"] < 2:
    p_selector = fit_prop_select
  else:
    p_selector = default_parent_selector
    
  klist_ga = GA_generator(klist_fit_func, kgen, GA_params, mutator = mutate, reproducer = cross,
    constraint_checker = kcheck, fitness_processes_per_eval=trials, num_elites=2, 
    parent_selector=p_selector, status=status, early_stop=early_stop)
    
  return klist_ga
  
def run_component_tests():

  g = nx.Graph()
  g.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(6,8),(6,9)])
  test_graph = eg.Experiment_Graph("test",nxgraph=g)
  print(test_graph.adj_mat)
  
  seed_set = np.zeros(10)
  seed_set[0] = 1
  
  LT_params = {"seed_set":seed_set, "a_threshold":2, "type":"absolute"}
  
  IC_params = {"seed_set":None, "c":.8}
  
  print("KList fitness on IC\n")
  klist_fit_func = klist_fit_wrapper(test_graph,10,test_graph.IC,IC_params,1)
  print(klist_fit_func([0],0,{}))
  klist_fit_func = klist_fit_wrapper(test_graph,10,test_graph.IC,IC_params,2)
  print(klist_fit_func([0,8],0,{}))
  print(klist_fit_func([0,8],0,{(0,8):-100}))
  
  print("\nNlist Fitness Tests on LT\n")
  nlist_fit_func = nlist_fit_wrapper(test_graph,10,test_graph.LT,LT_params,2)
  print(nlist_fit_func([0],0,{}))
  print(nlist_fit_func([0,2,4,8],0,{}))
  print(nlist_fit_func([0,2,4,8],.5,{}))
  print(nlist_fit_func([0,2,4,8],1,{}))
  print(nlist_fit_func([0,8],0,{(0,8):-100}))
 
  rand_mutate = nlist_mutate(test_graph,2,mutate_type="random",degree_wtd=False)
  neighbor_mutate = nlist_mutate(test_graph,2,mutate_type="neighbor",degree_wtd=False)
  distant_mutate = nlist_mutate(test_graph,2,mutate_type="distance",degree_wtd=False)
  
  ind = [1,5]
  print("\nNlist Mutate Tests Unweighted:",ind,"\n")
  for x in range(10):
    mutate_tests = [rand_mutate(ind,1.0),neighbor_mutate(ind,1.0),distant_mutate(ind,1.0)]
    line_length = 10
    pad1 = "   "*(line_length - len(mutate_tests[0]))
    pad2 = "   "*(line_length - len(mutate_tests[1]))
    mutate_tests.insert(1,pad1)
    mutate_tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*mutate_tests))
    

  ind = [1,5]
  k = len(ind)
  rand_mutate = nlist_mutate(test_graph,k,mutate_type="random",degree_wtd=True)
  neighbor_mutate = nlist_mutate(test_graph,k,mutate_type="neighbor",degree_wtd=True)
  distant_mutate = nlist_mutate(test_graph,k,mutate_type="distance",degree_wtd=True)
  
  print("\nNlist Mutate Tests Weighted:",ind,"\n")
  print(test_graph.get_deg_seq(relative=True))
  for x in range(10):
    mutate_tests = [rand_mutate(ind,1.0),neighbor_mutate(ind,1.0),distant_mutate(ind,1.0)]
    line_length = 10
    pad1 = "   "*(line_length - len(mutate_tests[0]))
    pad2 = "   "*(line_length - len(mutate_tests[1]))
    mutate_tests.insert(1,pad1)
    mutate_tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*mutate_tests))
    
    
  rand_mutate = klist_mutate(test_graph,k,mutate_type="random",degree_wtd=False)
  neighbor_mutate = klist_mutate(test_graph,k,mutate_type="neighbor",degree_wtd=False)
  distant_mutate = klist_mutate(test_graph,k,mutate_type="distance",degree_wtd=False)
  
  ind = [1,5]
  print("\nKlist Mutate Tests Unweighted:",ind,"\n")
  for x in range(10):
    mutate_tests = [rand_mutate(ind,1.0),neighbor_mutate(ind,1.0),distant_mutate(ind,1.0)]
    line_length = 10
    pad1 = "   "*(line_length - len(mutate_tests[0]))
    pad2 = "   "*(line_length - len(mutate_tests[1]))
    mutate_tests.insert(1,pad1)
    mutate_tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*mutate_tests))    
    
  rand_mutate = klist_mutate(test_graph,k,mutate_type="random",degree_wtd=True)
  neighbor_mutate = klist_mutate(test_graph,k,mutate_type="neighbor",degree_wtd=True)
  distant_mutate = klist_mutate(test_graph,k,mutate_type="distance",degree_wtd=True)
  
  ind = [1,5]
  print("\nKlist Mutate Tests Weighted:",ind,"\n")
  print(test_graph.get_deg_seq(relative=True))
  for x in range(10):
    mutate_tests = [rand_mutate(ind,1.0),neighbor_mutate(ind,1.0),distant_mutate(ind,1.0)]
    line_length = 10
    pad1 = "   "*(line_length - len(mutate_tests[0]))
    pad2 = "   "*(line_length - len(mutate_tests[1]))
    mutate_tests.insert(1,pad1)
    mutate_tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*mutate_tests))
    

  ksingle = k_list_crossover(test_graph, 3, 1.0, crossover_type="single_point", degree_wtd=True)
  kdegree = k_list_crossover(test_graph, 3, 1.0, crossover_type="degree_cooperation", degree_wtd=True)
  kmask = k_list_crossover(test_graph, 3, 1.0, crossover_type="network_mask", degree_wtd=True)    
  inda = [1,3,5]
  indb = [5,6,9]
  k = len(inda)
  print("\nKlist Crossover Tests Weighted:",inda,indb,"\n")
  for x in range(10):
    tests = [ksingle(inda,indb),kdegree(inda,indb),kmask(inda,indb)]
    pad1 = "\t"
    pad2 = "\t"
    tests.insert(1,pad1)
    tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*tests))
    
  nsingle = n_list_crossover(test_graph,k, 1.0, crossover_type="double_point", degree_wtd=True)
  ndegree = n_list_crossover(test_graph,k, 1.0, crossover_type="degree_cooperation", degree_wtd=True)
  nmask = n_list_crossover(test_graph,k, 1.0, crossover_type="network_mask", degree_wtd=True)    
  inda = [1,3,5]
  indb = [5,6,9]
  print("\nNlist Crossover Tests Weighted:",inda,indb,"\n")
  for x in range(10):
    tests = [nsingle(inda,indb),ndegree(inda,indb),nmask(inda,indb)]
    pad1 = "\t"
    pad2 = "\t"
    tests.insert(1,pad1)
    tests.insert(3,pad2)
    print("{} {} {} {} {}".format(*tests))
    

if __name__ == "__main__":
  main()
  run_component_tests()
  run_aggregate_tests()