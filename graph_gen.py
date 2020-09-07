import numpy as np

import experiment_graph as eg

# This script generates three types of graphs, all of which are returned
#  as Experiment_Graph objects:
# Barabasi albert graphs
#   ba_graph(N : # nodes,m : 1/2 average degree)
# Classic SBM
#   gen_sbm(N: # nodes, K: # communities,a,b, probabilities=False)
#      a : intra community link probability OR average intra community degree
#      b : inter community link probability OR average inter community degree
# Degree-Corrected SBM
#   gen_dc_sbm(N,K,a,b,gamma,kmin,kmax,probabilities=False)
#      gamma - float, degree exponent of powerlaw used to generate degree distributions
#      kmin - minimum degree when generating powerlaw distribution (normalized later,
#             doesnt determine final minimum degree in graph)
#      kmax - maximum degree when generating powerlaw distribution (normalized later,
#             doesnt determine final minimum degree in graph)
# NOTE: These were only intended to be used for graphs where K is a factor of N. If K doesnt
#       evenly divide N, then you will have to adjust the code to correctly bound the community
#       assignments.

# N = Size of graph in nodes
# K = num of communities
# a = intra community link probability OR average intra community degree
# b = inter community link probability OR average inter community degree
def gen_sbm(N,K,a,b,probabilities=False):

  am = np.random.uniform(size=(N,N))
  C = int(N/K)
  
  if not probabilities:
    a = a/C
    b = b/(N-C)
 
  for r in range(K):
    for s in range(r+1):
      slice = am[r*C:(r+1)*C,s*C:(s+1)*C]
      if r == s:
        am[r*C:(r+1)*C,s*C:(s+1)*C] = np.where(slice < a, 1.0, 0.0)
      else:
        am[r*C:(r+1)*C,s*C:(s+1)*C] = np.where(slice < b, 1.0, 0.0)
  am = np.tril(am,-1) + np.tril(am,-1).T
  return eg.Experiment_Graph("sbm",am=am)

# N = Size of graph in nodes
# K = num of communities
# a = intra community link probability OR average intra community degree
# b = inter community link probability OR average inter community degree
def gen_dc_sbm(N,K,a,b,gamma,kmin,kmax,probabilities=False):

  am = np.random.uniform(size=(N,N))
  C = int(N/K)
  
  if not probabilities:
    a = a/C
    b = b/(N-C)
  
  #print(np.around(am,1))
  
  node_popularities = []
  for r in range(K):
    comm_degs = powerlaw_degrees(gamma,kmin,kmax,C)
    sum_deg = sum(comm_degs)
    node_popularities += [cd/sum_deg for cd in comm_degs]
    #alpha = (a*C)/sum(node_popularities)
  node_popularities = np.array(node_popularities)
  #node_popularities = node_popularities.reshape(1,N)
  #print(node_popularities)
  
  targets = np.outer(node_popularities,node_popularities)
  
  #print(np.around(am,1)) 
  C2 = C**2
 
  for r in range(K):
    for s in range(r+1):
      rand_slice = am[r*C:(r+1)*C,s*C:(s+1)*C]
      target_slice = targets[r*C:(r+1)*C,s*C:(s+1)*C]
      target_sum = np.sum(target_slice)
      if r == s:
        wrs_u = a*C2/target_sum
        target_slice *= wrs_u
        am[r*C:(r+1)*C,s*C:(s+1)*C] = np.where(rand_slice < target_slice, 1.0, 0.0)
      else:
        wrs_u = b*C2/target_sum
        target_slice *= wrs_u
        am[r*C:(r+1)*C,s*C:(s+1)*C] = np.where(rand_slice < target_slice, 1.0, 0.0)
  am = np.tril(am,-1) + np.tril(am,-1).T
  #print(np.around(am,1))
  return eg.Experiment_Graph("sbm",am=am)
  
    
def ba_graph(N,m):
  
  am = np.zeros((N,N))
  link_draw = m
  total = 0
  for n in range(1,N):
    if m%1 != 0.0:
      link_draw = 1 + np.random.poisson(m-1)

    links = int(min(n,link_draw))
    total += links
    probs = np.sum(am,axis=0)[0:n]
    if np.sum(probs) == 0:
      probs = np.ones(n) / n
    else:
      probs = probs / np.sum(probs)
    
    # np.arange non-inclusive,choice is by default non-replacement
    nbrs = np.random.choice(np.arange(n),size=links,replace=False,p=probs)
    # print(n,links,nbrs)

    # if n in nbrs or links != len(nbrs):
      # print("HERE")
    am[n,nbrs] = 1
    am[nbrs,n] = 1
    # print(am)
  
  name = "BA:" + str(N) + "," + str(m)
  new_ba = eg.Experiment_Graph(name,am=am)
  # print(total,np.sum(am))
  return new_ba
    

def sbm_gen_params(N,groups,avg_deg_intra,avg_deg_inter):

  sbm_membership = [int(i/(N/groups)) for i in range(N)]
  inter_edges = avg_deg_inter * (N / groups)
  intra_edges = 2 * avg_deg_intra * (N / groups)
  inter_comm_edges = np.full((groups,groups),inter_edges)
  np.fill_diagonal(inter_comm_edges,intra_edges)
  return [sbm_membership, inter_comm_edges]
  
def powerlaw_degrees(gamma,kmin,kmax,N):
  pdf = np.array([x**-gamma for x in range(kmin,kmax+1)])
  pdf = pdf / np.sum(pdf)
  return np.random.choice(range(kmin,kmax+1),size=N,p=pdf)
  