import networkx as nx
import numpy as np
import random as r
import math
import collections
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import zeta
import traceback

import scipy.stats
import scipy.special as spesci
from scipy.stats import truncnorm

## CLASS ##
## CONSTRUCTION ## 
# sbm(name,edge_dist,group_dist=None,standard_grps=None): 
#    name: String - any desired label
#    edist - function(int, int, dictionary) -> float [0,1]; function returns the affiliation between
#            two communities given a dictionary keyed by node specifying community membership
#            e.g. def affiliated(i,j,temp_comms): 
#                   if i==j:  
#                     return 1.0
#                   return 0.0
# 
#    gdist - function(int) -> int in {0,Graph Size N}, function which generates community sizes given 
#            a graph size.  e.g. def tenths(X): return int(X/10)
#    standard_groups - list(float in [0,1]), sums to 1, alternate way to determine community sizes

## USE ##
# sbm objects are meant to flexibly generate any manner of probabilistic community sizes and affiliations
# then, the object can be used to label any given networkx graph as follows:
#   my_sbm.apply(example_graph)

# Community assignments can later be accessed directly from the graph, as:
#   example_graph.graph[my_sbm.name] = len(self.grps)
#   example_graph.nodes[i]['communities']  - the communities i belongs to, 
#                                             which are of the form: my_sbm.name + community index
#   example_graph.nodess[j]['comm_acq'] - the communities for each edge between j and all its neighbors, 
#   example_graph[i][j]['communities'] - the communities through which i and j share an edge

# One can apply multiple sbms to a single graph, allowing for a rich range of expression, for example:
#   jobs_sbm.apply(local_community_graph)
#   hobbies_sbm.apply(local_community_graph)

class sbm:            

  def __init__(self,name,edge_dist,group_dist=None,standard_grps=None):      
    self.name = name
    self.edist = edge_dist
    self.gdist = group_dist
    self.grps = standard_grps
  
  # IN: graph - networkx graph object
  # OUT: None
  # Applies the SBM object's edge and group distributions to the graph,
  # storing community assignments in the networkx graph's node and edge dictionaries
  def apply(self,graph):
    temp_comms = self.apply_communities(graph)
    if temp_comms == None:
      print("Error occured during class labeling.")
      return None
    self.apply_edges(graph,temp_comms)
      
      
  # IN: graph - networkx graph object
  # OUT: dictionary[int] -> int, community assignments for all nodes
  def apply_communities(self,graph):
  
    if self.gdist == None and self.grps == None:
      print("Error: undefined group distribution.")
      return None
      
    temp_communities = {}
    
    # Traditional SBM formulation, each comm has a prior in gdist
    if self.gdist == None:
      for n in graph.nodes:
        if graph.nodes[n].get('communities',None) == None:
          graph.nodes[n]['communities'] = set()
        sum = 0.0
        n_r = r.random()
        for x in range(self.grps):
          sum += self.grps[x]
          if n_r < sum:
            comm_name = self.name + "-" + str(x)
            smart_dict_append(graph.nodes[n],'communities',comm_name)
            smart_dict_append(temp_communities,x,n)

      graph.graph[self.name] = len(self.grps)
      smart_dict_append(graph.graph,'communities',self.name)
      
    # Alternate formulation, groups sizes are determined by gdist(V)
    # and randomly sampled to fill.  Will not be a "true" sample of gdist
    # and deteriorates as gdist(V) ~> order of V, as it will run out of nodes
    # to fill the final group.
    else:
      V = graph.order()
      unalloc_nodes = set(graph.nodes)
      x = 0
      while len(unalloc_nodes) > 0:
        x += 1
        comm_size = self.gdist(V)
        comm_size = min(comm_size, len(unalloc_nodes))
        comm_x = r.sample(unalloc_nodes,comm_size)
        unalloc_nodes.difference_update(comm_x)
        cx_name = self.name + "-" + str(x)
        for cx_n in comm_x:
          smart_dict_append(graph.nodes[cx_n],'communities',cx_name)
          smart_dict_add(graph.nodes[cx_n],'comm_acq',cx_name)
          smart_dict_append(temp_communities,x,cx_n)
          
      graph.graph[self.name] = x
      smart_dict_append(graph.graph,'communities',self.name)
      return temp_communities
      
  # IN: graph - networkx graph
  #     temp_comms - dictionary partition of nodes to communities
  # OUT: None
  # graph mutated with added edges and stored community assignments
  def apply_edges(self,graph,temp_comms):
  
    omega_ij = {}
    for i in temp_comms:
      for j in temp_comms:
        if j > i:
          continue
        omega_ij[i,j] = self.edist(i,j,temp_comms)
        omega_ij[j,i] = omega_ij[i,j]

    for i in graph.nodes:
      icom = self.extract_index(graph.nodes[i]['communities'])
          
      for j in graph.nodes:
        if i <= j:
          continue
        jcom = self.extract_index(graph.nodes[j]['communities'])
      
        if r.random() < omega_ij[icom,jcom]:
          if not graph.has_edge(i,j):
            graph.add_edge(i,j)
          smart_dict_append(graph[i][j],'communities',self.name)

          if icom != jcom:
            graph.nodes[i]['comm_acq'].add(self.name + "-" + str(jcom))
            graph.nodes[j]['comm_acq'].add(self.name + "-" + str(icom))
                
  def extract_index(self,communities):
    for comm in communities:
      if self.name in comm:
        return int(comm.split("-",1)[1])
        
# IN: graph - networkx graph object
#     name - string for community name
#     communities - list(int) length |V| of graph
# OUT: None
# Labels a graph, without use of an sbm object, that already has edges
# according to the community assignments in communities
def label_graph(graph, name, communities):
  
  for n in graph.nodes():
    # if graph.nodes[n].get('communities',None) == None:
      # graph.nodes[n]['communities'] = set()
    comm_ind = communities[n]
    comm_name = name + "-" + str(comm_ind)
    smart_dict_append(graph.nodes[n],'communities',comm_name)
    smart_dict_add(graph.nodes[n],'comm_acq',comm_name)
    
  graph.graph[name] = len(np.unique(communities))
  smart_dict_append(graph.graph,'communities',name)
  
  for e in graph.edges():
    i,j = e
    icom = extract_ind(graph.nodes[i]['communities'],name)
    jcom = extract_ind(graph.nodes[j]['communities'],name)
        
    smart_dict_append(graph[i][j],'communities',name)

    if icom != jcom:
      graph.nodes[i]['comm_acq'].add(name + "-" + str(jcom))
      graph.nodes[j]['comm_acq'].add(name + "-" + str(icom))
        
# IN: graphs - list(networkx graphs), each must have an sbm applied
#     (scramble) - Boolean, toggle whether to decouple node indices between graphs
# OUT: networkx graph, with the union of all sbm information from all in graphs 
def merge_graphs(graphs,scramble=True):

  sum_g = nx.Graph()
  for g in graphs:
    nmap = list(range(g.order()))
    if scramble:
      r.shuffle(nmap)
      
    name = g.graph['communities'][0]
    sum_g.graph[name] = g.graph[name]
    smart_dict_append(sum_g.graph,'communities',name)
    
    for n_og in g.nodes():
      n = nmap[n_og]
      sum_g.add_node(n)
      n_comm = g.nodes[n_og]['communities'][0]
      smart_dict_append(sum_g.nodes[n],'communities',n_comm)
      for ca in g.nodes[n_og]['comm_acq']:
        smart_dict_add(sum_g.nodes[n],'comm_acq',ca)
    
    for e in g.edges():
      i_og,j_og = e
      i = nmap[i_og]
      j = nmap[j_og]
      sum_g.add_edge(i,j)
      smart_dict_append(sum_g[i][j],'communities',name)
      
  return sum_g
    
 
def extract_ind(communities,name):
  for comm in communities:
    if name in comm:
      return int(comm.split("-",1)[1])
        
def pull_index(communities,name):
  for comm in communities:
    if name in comm:
      return int(comm.split("-",1)[1])

def smart_dict_append(d,k,val):
  if d.get(k) == None:
    d[k] = [val]
  else:
    d[k].append(val) 

def smart_dict_add(d,k,val):
  if d.get(k) == None:
    d[k] = set([val])
  else:
    d[k].add(val) 

# IN: graph - networkx graph with community assignments as from sbm object or label_graph
#     (cmap) - matplotlib color map
#     (communities) - list(string), which communities to consider when drawing the graph
#     (title) - string, title for matplotlib drawing and, if saving figure, the filename
#     (save) - Boolean, toggle whether figure is saved
#     (show) - Boolean, toggle whether figure is shown
#     (pos) - networkx position of nodes for drawing
# OUT: None
# Using matplotlib and networkx, visualizes the graph with nodes colored by community assignment
# and edges by community (a single color for all inter-community edges, and the community color
# for intra-community edges)
def draw_community_graph(graph,cmap="tab10",communities="all",title="Default",save=False,show=False,pos=None):
  plt.clf()
    
  if communities == "all":
    communities = graph.graph['communities']
  
  num_c = len(communities)
  
  if num_c > 10 and num_c <= 20:
    cmap = "tab20"
  
  cmap = plt.get_cmap(cmap)
  colors = cmap(range(num_c))
  color_dict = {}
  for x in range(num_c):
    color_dict[communities[x]] = colors[x]
    
  inter_graph = nx.MultiGraph()
  intra_graph = nx.MultiGraph()
  
  for comm in communities:
    for e in graph.edges():
      if comm in graph[e[0]][e[1]]['communities']:
        icom = pull_index(graph.nodes[e[0]]['communities'],comm)
        jcom = pull_index(graph.nodes[e[1]]['communities'],comm)
        if icom != jcom:
          inter_graph.add_edge(e[0],e[1],color=color_dict[comm])
        else:
          intra_graph.add_edge(e[0],e[1],color=color_dict[comm])
  
  inter_colors = []
  for e in inter_graph.edges(data='color'):
    inter_colors.append(e[2])
  intra_colors = []
  for e in intra_graph.edges(data='color'):
    intra_colors.append(e[2])
    
  if pos == None:
    pos = nx.spring_layout(graph)
  #pos = nx.spring_layout(intra_graph)
  nx.draw(intra_graph,pos,edge_color=intra_colors,width=.5,node_size=50, node_color="gray")
  nx.draw(inter_graph,pos,edge_color=inter_colors,style='dotted',width=1.0,node_size=50, node_color="gray")
  if save:
    filename = "Graph Figures/" + title
    plt.savefig(filename + '.png', bbox_inches='tight')
  if show:
    plt.show()
    
  return pos
  
  # V : size of graphs built
  # SBMs : list of SBM objects 
  # n : desired number of graphs built with these parameters
  
  # out: CLI -> V, E, <num comm interactions / node>
  #      IMG -> graph with colored + numbered edges?
  #      Obj -> [nx.graphs]
  #             edges have attribute 'community' in SBMs.name + _Index
  #             nodes have attribute 'comm_acq' = | edge communities |
  #             nodes have attribute 'communities' = { node communities }
  
  
## Some example edge distribution and group distributions follow ##

# Edge_Dist functions 
def assortative(a,d):
  
  def internal(i,j,t_comms):
    if i == j:
      return a
    return d
  
  return internal
  
# in a, d
# scales d by the maximum of either group i/j cardinality s.t.
# expected degree from smaller group node to all of larger group
# is equal to d.  e.g., if d = 3, each small group node will have on average  
# 3 neighbors in the larger group.
def scaled_assortative(a,d):
  
  def internal(i,j,t_comms):
    if i == j:
      return a
    i_size = len(t_comms[i])
    j_size = len(t_comms[j])
    size = max(i_size,j_size)
    return d/size
  
  return internal
  
def decay_distance(a,d,factor):
  def internal(i,j,t_comms):
    if i == j:
      return a
    return d*np.exp(-(abs(j-i)-1)*factor)
  
  return internal

# Group_Dist functions

# in: n, std
# out: integer sampled from truncated norm(n,std) in [0,2n]
def groups_size_n(n,std): 
  
  def internal(V):
    # print(V,n,std)
    tnorm = get_truncated_normal(n,sd = std, low=0,upp=2*n)
    return int(tnorm.rvs())
  
  return internal
  
def groups_exact_n(n):
  
  def internal(V):
    return n
  
  return internal
    
  
def groups_poisson(l,scale=1):
  
  def internal(V):
    return np.random.poisson(l)
  
  return internal

# in: k desired number of expected groups
# out: integer sampled from truncated norm, centered V/k, std = v/(3k), in [0,2V]
def k_groups_var(k):

  def internal(V):
    tnorm = get_truncated_normal(V/k,sd = V/(3*k), low=0,upp=2*(V/k))
    return int(tnorm.rvs())
  
  return internal
  
def groups_powerlaw(alpha,max):
  
  def internal(V):
    ra = r.random()
    cdf = cdf_exp(1,max,alpha)
    scale = 1.0/cdf[-1]
    cdf = [c*scale for c in cdf]
    for x in range(len(cdf)):
      if ra < cdf[x]:
        return x+1
  
  return internal  
  
def g_plaw_norm(alpha,max,n,std):
  
  def internal(V):
    b = groups_powerlaw(alpha,max)(V)
    a = groups_size_n(n,std)(V)
    return a+b
  
  return internal
  
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)  


#Input: minimum degree, max degree, and an exponent alpha
#Output: returns a CDF of the exponential distribution starting at kmin according to Alpha, length (kmax -kmin)
def cdf_exp(kmin, kmax, alpha):
	
	c = 1 / reimann_start(alpha, kmin)
	
	cdf = []
	total = 0
	for x in range(kmin, kmax+1):
		total += c*(x ** (-1 * alpha))
		cdf.append(total)
		
	return cdf
	
def reimann_stop(alpha, stop):

	sum = 0.0
	for x in range(1,stop + 1):
		sum = sum + (1 / ((x*1.0) ** alpha))
	
	return sum
	
def reimann_start(alpha, start):

	if start == 1:
		return zeta(alpha)
	elif start < 1:
		print("Zeta called on non positive start value.")
		return 0
	else:
		return zeta(alpha) - reimann_stop(alpha, start-1)
