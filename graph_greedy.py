import networkx as nx
import numpy as np
import random as r
import math
import collections
import copy
import time
import sys

import experiment_graph as eg

# IN: graph - Experiment_Graph
#     fun - function(networkx graph, list(node indices)) -> float
#     k - list(int), sizes of solutions desired
#     (submodular) - Boolean, if true will use the CELF optimization
#     (progress) - Boolean, if true will print progress statements during execution
# OUT: results - list(lists(int)), set of solutions, one for each size in k
#      evals - integer, number of times fun invoked
def greedy(graph,fun,k,submodular=False,progress=False):

  if submodular:
    return greedy_sub(graph,fun,k,progress)
    
  return greedy_general(graph,fun,k,progress)
  
def greedy_sub(graph,fun,k,progress=False):

  k.sort()
  
  greedy = set()
  current = set()
  last = []
  k_ind = 0
  results = []
  
  ngraph = graph.nxgraph
  
  # last: [[v1, i], [v2, j],...,[vn,n]] vk >= vj if k < j
  for n in ngraph.nodes:
    val = fun(set([n]),graph)
    insort(last,val,n)
  
  if progress:
    k_start = time.time()
    sys.stdout.write("\rK:{}/{}".format(k[k_ind], k[-1]))
  evals = graph.N
  
  while len(greedy) < k[-1]:
      
    greedy_val = fun(greedy,graph)
    if len(last) == 0:
      while len(results) < len(k):
        results.append(greedy.copy())
      #print("Graph smaller than requested set size.")
      return results, evals
    max_increment = 0.0
    index = len(last)-1
    
    check = True
    while last[index][0] > max_increment or check:
      current = greedy.copy()
      candidate = last.pop(index)
      n = candidate[1]
      current.add(n)
      evals += 1
      val = fun(current,graph) - greedy_val
      insort(last,val,n)
      if val > max_increment:
        check = False
        max_increment = val
        top_n = n
        top_index = index
      index -= 1
      if index < 0:
        break
    
    if check:
      greedy.add(last[-1][1])
      last.pop()
    else:
      greedy.add(top_n)
      last.pop(top_index)
    
    if len(greedy) == k[k_ind]:
      results.append(greedy.copy())
      k_ind += 1
      if progress and k_ind < len(k):
        k_end = time.time()
        sys.stdout.write("\rK:{}/{} Last K in {}s"
          .format(k[k_ind], k[-1], np.round(k_end - k_start,3)))
        sys.stdout.flush()
        k_start = time.time()
    
  return results, evals
  
def greedy_general(graph,fun,k,progress=False):

  k.sort()
  
  greedy = set()
  current = set()
  k_ind = 0
  results = []
  ngraph = graph.nxgraph
  available = set(ngraph.nodes)
  evals = 0
  
  if progress:
    k_start = time.time()
    sys.stdout.write("\rK:{}/{}".format(k[k_ind], k[-1]))
  
  while len(greedy) < k[-1]:  
  
    if len(available) == 0:
      while len(results) < len(k):
        results.append(greedy.copy())
      #print("Graph smaller than requested set size.")
      return results, evals
    greedy_val = fun(greedy,graph)
    evals += 1
    max_increment = 0.0
    count = 0
    
    check = True
    for n in available:
      count += 1
      
      current = greedy.copy()
      current.add(n)
      val = fun(current,graph) - greedy_val
      evals += 1
      if val > max_increment:
        check = False
        max_increment = val
        top_n = n
        
      if progress:
        sys.stdout.write("\rK:{}/{} Iteration on {}/N"
          .format(len(current), k[-1], count))
        sys.stdout.flush()
        
    if check:
      greedy.add(available.pop())
    else:
      greedy.add(top_n)
      available.remove(top_n)
    
    if len(greedy) == k[k_ind]:
      results.append(greedy.copy())
      k_ind += 1
      
      if progress and k_ind < len(k):
        k_end = time.time()
        sys.stdout.write("\rK:{}/{} Last K in {}s"
          .format(k[k_ind], k[-1], np.round(k_end - k_start,3)))
        sys.stdout.flush()
        k_start = time.time()
    
  return results, sum([(graph.N-x) for x in range(0,k[-1])])

def insort(list,val,n):
  top = len(list)-1
  bot = 0
  mid = int((top-bot)/2)
  
  if len(list) <= 2:
    mid = 0
    for x in range(len(list)):
      mid = x
      if list[x][0] > val:
        break
    list.insert(mid,[val,n])
    return
      
  while top-bot > 1:
    if list[mid][0] > val:
      top = mid
    elif list[mid][0] < val:
      bot = mid
    else:
      break
    mid = bot+int((top-bot)/2)
  
  if list[mid][0] > val:
    list.insert(mid,[val,n])
  elif list[mid+1][0] < val:
    list.insert(mid+2,[val,n])
  else:
    list.insert(mid+1,[val,n])

def myfun(s,graph):
  contact = set()
  for x in s:
    contact.update(set(graph.nxgraph.neighbors(x)))
  return len(contact)
  
def main():
  print("Running Graph Greedy as main, tests follow:")
  
  graph = nx.barabasi_albert_graph(100,4)
  test_graph = eg.Experiment_Graph("test",nxgraph=graph)
  
  def num_nbrs(group,graph):
    neighbors = [[i for i in graph.nxgraph.neighbors(ind)] for ind in group]
    total_unique = {nbr for nblist in neighbors for nbr in nblist}
    return len(total_unique)
    
  res = greedy(test_graph,num_nbrs,[3,4,5],submodular=False,progress=True)
  print("Supermodular",res)
  
  res = greedy(test_graph,num_nbrs,[3,4,5],submodular=True,progress=True)
  print("Submodular",res)
  
if __name__ == "__main__":
  main()
