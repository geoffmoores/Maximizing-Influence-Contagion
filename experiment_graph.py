import sys
import time
import numpy as np
import random as r
import networkx as nx
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.special as spesci
import copy
import igraph as ig
import graph_tool.all as gt
import re
from os import path as ospath
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import pickle
from sklearn.metrics import mutual_info_score
import scipy.stats


# This is a class with a host of functionality used across the thesis

## CONSTRUCTION ##
# Experiment_Graph(type, read_data=[], am=np.array([]), nxgraph=None, gtgraph=None,
#                   filename="default_filename", path="", gname="default")
#    type: String - any label for the type of graph

#  The following four optional parameters provide all network data for construction 
#  of the Experiment_Graph object. If multiple are given, the first in this order is preferred.
#  If none is provided, a warning of an empty graph is printed.
#    read_data: list of lists, [edge_list, weights]
#               - edge_list  : list of tuples, node pairs with an edge between them
#               - weights    : list of integers / floats, weights[i] = weight of edge_list[i] 
#    am       : numpy 2D array of edge weights       
#    nxgraph  : networkx .Graph type                
#    gtgraph  : graph-tool .Graph type

#    filename : String, filename for storing the object as a pickle
#    path     : String, directory to store pickle
#    gname    : String, name of graph

## FIELDS ##
# Primarily it hosts a CSR Sparse Matrix representation of a graph : .adj_mat
# and three distinct graph versions for use of separate libraries:
#   .gtgraph  : graph_tool 
#   .igraph   : igraph     
#   .nxgraph  : networkx   

#  .N = # vertices
#  .M = # edges
#  .gname = String, name of graph
#  .edges, .wts = lists, only set if reading the data from a list
#  .discrete = Boolean, edge weights discrete or not
#  .weighted = Boolean, edges weighted or not
#  .rec_type = String, classification parameter for graph-tool SBM recovery
#  .results = Dictionary, used to store experimental results with the graph in question
#  .type = String label, used to define real-world network categories for thesis
#  .filename = String for filename to pickle the Experiment_Graph object
#  .path = path to put the pickle when saving 
#  .structures = Dictionary, holds community structures / other partitions found

#  .temp = Dictionary, for convenient storage of any metadata associated with the graph

class Experiment_Graph:
  
  def __init__(self,type,read_data=[],am=np.array([]),nxgraph=None,gtgraph=None,filename="default_filename",path="",gname="default"):
    
    # Different methods of receiving graph data
    if len(read_data) > 0: 
      self.edges = read_data[0]
      self.wts = read_data[1]
      N = np.amax(self.edges)+1
      self.adj_mat = csr_matrix((self.wts,(self.edges)),shape=(N,N))
      self.adj_mat.sum_duplicates()
      self.adj_mat += self.adj_mat.transpose()
    elif nxgraph != None: # This will hog memory on very large graphs, it is provided mostly
      self.adj_mat = csr_matrix(nx.to_numpy_array(nxgraph)) #  for testing convenience with nx
      self.nxgraph = nxgraph
    elif gtgraph != None:
      self.adj_mat = self.adj_from_gt(gtgraph)
    else:
      self.adj_mat = csr_matrix(am)
      if am.shape[0] == 0:
        print("Warning, created empty Experiment graph.")
        
    self.N = self.adj_mat.shape[0]
    self.gname = gname
    
    temp = self.adj_mat.tolil()
    temp.setdiag(0)
    self.adj_mat = temp.tocsr()
    self.adj_mat.eliminate_zeros()
    self.M = self.adj_mat.getnnz()/2
   
    self.weighted = True
    if len(np.unique(self.adj_mat.data)) == 1:
      self.weighted = False
      if self.adj_mat.data[0] != 1.0: # to make some things cleaner, this just replaces all weights with 1 if there is only one edge weight
        self.adj_mat.data = np.ones(len(self.adj_mat.data))
      
    self.discrete = False
    diff = self.adj_mat.floor() - self.adj_mat
    if diff.count_nonzero() == 0:
      self.discrete = True
      
    if len(np.where(self.adj_mat.data < 0)[0]) > 0:
      self.rec_type = "real-normal"
    elif self.discrete:
      self.rec_type = "discrete-poisson"
    else:
      self.rec_type = "real-exponential"
      
    if nxgraph == None:
      self.nxgraph = self.nx_from_adj(self.adj_mat)
    self.igraph = self.ig_from_adj(self.adj_mat)
    self.gtgraph = self.gt_from_adj(self.adj_mat)
    #self.pos = nx.spring_layout(self.nxgraph)

    self.results = {}
    
    self.type = type
    self.filename = filename
    self.path = path
    self.structures = {}
    
    self.temp = {}
    
    
  # IN: am - csr_matrix
  # OUT: None
  # sets the library specific graph objects according to am
  def set_graphs_adj(self,am):
    self.nxgraph = self.nx_from_adj(am)
    self.igraph = self.ig_from_adj(am)
    self.gtgraph = self.gt_from_adj(am)
    
  # IN: am - csr_matrix 
  # OUT: None
  # .gtgraph populated from adj_mat
  def gt_from_adj(self,adj_mat):
    N = adj_mat.shape[0]
    edge_list = adj_mat.nonzero()
    weights = adj_mat.data
    g = gt.Graph()
    g.add_vertex(N)
    g.add_edge_list(np.transpose(edge_list))
    
    ew = g.new_edge_property("double")
    ew.a = weights
    g.ep['weight'] = ew
    
    return g

  # IN: am - csr_matrix 
  # OUT: None
  # .nxgraph populated from am
  def nx_from_adj(self,am):
    N = am.shape[0]
    edges = np.transpose(am.nonzero())
    wts = am.data
    efrom = [(edges[i][0],edges[i][1],wts[i]) for i in range(len(wts))]
    nxgraph = nx.Graph()
    nxgraph.add_nodes_from(range(N))
    nxgraph.add_weighted_edges_from(efrom)
    return nxgraph
    
  # IN: am - csr_matrix 
  # OUT: None
  # .igraph populated from am
  def ig_from_adj(self,am):
    ig_graph = ig.Graph()
    N = am.shape[0]
    ig_graph.add_vertices(N)
    ig_graph.add_edges(np.transpose(am.nonzero()))
    ig_graph.es['weight'] = am.data 
    return ig_graph

  # IN: g - graph-tool Graph object
  # OUT: csr_matrix, adjacency matrix data from g
  # Warning: this will memory hog as is, but currently not used
  # TODO: adjust for efficient usage of csr_matrix
  def adj_from_gt(self,g):
    N = g.num_vertices()
    am = np.zeros((N,N))
    if "weight" in g.edge_properties:
      edges = g.get_edges(eprops=[g.ep.weight])
      for edge in edges:
        am[int(edge[0]),int(edge[1])] = edge[2]
    else:
      edges = g.get_edges()
      for edge in edges:
        am[int(edge[0]),int(edge[1])] = 1.0
        am[int(edge[1]),int(edge[0])] = 1.0
    return csr_matrix(am)
  
  
  # IN:  (alpha) - Real number [0,1], proportion of edges to rewire
  #      (ret_Exp_Graph) - Boolean, whether or not to return a new object or simply mutate the calling object
  # OUT: Experiment_Graph object - if ret_Exp_Graph == True, returns a new object
  
  # Generates a configuration graph from self, a proportion of edges equal to alpha are randomly 
  # rewired (with replacement)
  # NOTE: Only works for unweighted graphs.  Should not be used by default.
  def config_graph(self,alpha=1.0,ret_Exp_Graph=False):
    new_graph = self.gtgraph.copy()
    if alpha == 1.0:
      gt.random_rewire(new_graph)
    else: 
      M = self.gtgraph.num_edges()
      rw_edge_count = int(M*alpha)
      gt.random_rewire(new_graph,n_iter=rw_edge_count,edge_sweep=False)
    if ret_Exp_Graph:
      am = self.adj_from_gt(new_graph)
      new_Exp_G = Experiment_Graph(nx.Graph(),type=self.type,am=am)
      return new_Exp_G
    return new_graph
    
    
  # IN: None
  # Out: None
  # adj_mat is normalized, then all edge weights are shifted such that
  # the minimum edge weight is 0. 
  def norm_pos_adj_mat(self):
    
    if not self.weighted:
      return None
      
    link_wts = self.adj_mat.data
    link_mean = np.mean(link_wts)
    link_std = np.std(link_wts)
    
    if link_std == 0:
      link_std = 1

    link_wts -= link_mean
    link_wts /= link_std
    
    link_wts -= np.min(link_wts)
    
    if "str_seq" in self.temp:
      del self.temp["str_seq"]
      
  # IN: None
  # Out: None
  # all edge weights are shifted such that the minimum edge weight is 0
  # and then all weights are divided by the maximum weight
  def scale_pos_adj_mat(self):
    
    if not self.weighted:
      return None
      
    link_wts = self.adj_mat.data
    
    if np.min(link_wts) < 0:
      link_wts -= np.min(link_wts)
      
    link_wts /= np.max(link_wts)
    
    if "str_seq" in self.temp:
      del self.temp["str_seq"]
    
    
  # IN: i - integer in [0,self.N-1] of node of interest
  # OUT: degree(v_i) - int
  # Caches degree sequence in temp["deg_seq"] 
  def get_degree(self,i):
    
    if "deg_seq" not in self.temp:
      self.temp["deg_seq"] = np.array([self.adj_mat.getcol(i).getnnz() for i in range(self.N)])
    i = int(i)
    if i < 0 or i > self.N-1:
      print("Warning: asked for degree of node",i,"not in graph of size",self.N)
      return 0
    return self.temp["deg_seq"][i]
    
  # IN: (relative) - Boolean, if true the returned array is divided by the total number of edges
  # OUT: degree sequence - 1D np.array (ints), the degree sequence of the graph
  # Caches result in temp["deg_seq"] 
  def get_deg_seq(self,relative=False):
    if "deg_seq" not in self.temp:
      self.temp["deg_seq"] = np.array([self.adj_mat.getcol(i).getnnz() for i in range(self.N)])
    if relative:
      return self.temp["deg_seq"]/self.adj_mat.getnnz()
    return self.temp["deg_seq"]
    
  # IN: i - integer in [0,self.N-1] of node of interest
  # OUT: strength(v_i) - float, the sum of edge weights
  # Caches strength sequence in temp["str_seq"] 
  def get_strength(self,i):
    
    if "str_seq" not in self.temp:
      self.temp["str_seq"] = np.array(self.adj_mat.sum(0))[0]
    i = int(i)
    if i < 0 or i > self.N-1:
      print("Warning: asked for strength of node",i,"not in graph of size",self.N)
      return 0
    return self.temp["str_seq"][i]
    
  # IN: (relative) - Boolean, if true the returned array is divided by the sum of edge weights
  # OUT: strength sequence - 1D np.array, the strength sequence of the graph
  # Caches result in temp["str_seq"] 
  def get_str_seq(self,relative=False):
    if "str_seq" not in self.temp:
      self.temp["str_seq"] = np.array(self.adj_mat.sum(0))[0]
    if relative:
      return self.temp["str_seq"]/self.adj_mat.sum()
    return self.temp["str_seq"]
    
  # IN: (debug) - Boolean, deprecated (no effect)
  #     (recalc) - Boolean, if False then the cached version of distances will be overwritten
  # OUT: 2D np.array, distances in hops between all node pairs in the graph (np.Inf if no path exists)
  # Stores result in self.distances
  def get_distances(self,debug=False,recalc=False):
    if "distances" in self.temp and not recalc:
      return self.distances
    
    delta = True
    
    undirected = self.adj_mat.transpose().maximum(self.adj_mat)
    undirected.data = np.where(undirected.data != 0, 1, 0)
    distances = undirected.todense()
    distances = np.where(distances == 0, np.Inf, distances)
    np.fill_diagonal(distances,0)
    
    steps = 1
    while delta:
      steps += 1
      reach = np.where(distances != np.Inf, distances, 0)*undirected
      reach = np.where(reach == 0,  np.Inf, steps)
      new_distances = np.where(distances > reach, reach, distances)
      if np.all(new_distances == distances) or np.Inf not in new_distances:
        delta = False
      distances = np.copy(new_distances)
      
    self.temp["distances"] = True
    self.distances = np.copy(distances)
    return self.distances
      
  # IN:  (alpha) - Real number [0,1], proportion of edges to rewire
  #      (ret_Exp_Graph) - Boolean, whether or not to return a new object or simply mutate the calling object
  # OUT: Experiment_Graph object - if ret_Exp_Graph == True, returns a new object
  
  # Generates a configuration graph from self, a proportion of edges equal to alpha are randomly 
  # rewired (with replacement). Preserves (in expectation) the degree and strength of all nodes.
  # This is an extension of the work of Palowitch et al "Significance-based community detection in weighted networks"
  def wtd_config_graph(self,alpha=1.0,ret_Exp_Graph=True):
  
    # In some cases we will do this twice, once for the negative edges, and once
    # for the positive edges. This cleans that up a bit.
    # INPUT: adjacency matrix > 0 forall ij, alpha in (0,1]
    # OUTPUT: adjacency matrix of Weighted Configuration model, scaled by Alpha
    def wtd_config_helper(in_am,alpha=1.0):

      def get_p_i(i):
        p_i = (d_i[i]/(Deg))*d_i
        return np.where(p_i > 1, 1, p_i)
        
      def get_f_i(i):
        f_i = (s_i[i]/Str)*s_i
        f_i = f_i / np.where(p_i == 0, 1.0, p_i)
        return f_i
      
      N = in_am.get_shape()[0]
      Deg = in_am.getnnz() # total degree
      Str = in_am.sum() # total strength 
      links = csr_matrix(in_am)
      links.data = np.ones(Deg) # sparse matrix of links
      s_i = np.array(in_am.sum(0))[0] # strength sequence
      d_i = np.array([in_am.getcol(i).getnnz() for i in range(N)]) # degree sequence
      
      if Str == 0 or Deg == 0:
        print("Provided matrix of 0 strength or 0 degree, cannot configure.")
        print(in_am.data)
        print(Deg, Str)
        return in_am
      
      # we first estimate k_hat as in eq. 4 of Palowitch
      k_num, k_den = [0,0]
      for i in range(N):
        
        p_i = get_p_i(i)
        f_i = get_f_i(i)
        num = links[i].multiply(in_am[i] - f_i)
        num = num.power(2).sum()
        k_num += num
        
        k_den += (f_i**2).sum()
      
      k_hat = k_num / k_den
      
      # We now have everything we need to build our new matrix
      new_am = lil_matrix((N,N))
      Epsilon_i = 1
      
      # Now we generate our new matrix with the model described in section 2.1
      # but additionally we scale p_i according to alpha, such that the new 
      # matrix is (in expectation) only alpha*(full configuration model)
      for i in range(1,N):
        
        p_i = get_p_i(i)
        f_i = get_f_i(i)

        if k_hat != 0:
          Epsilon_i = np.random.gamma(1/k_hat,k_hat,i)
        
        p_i *= alpha
        # print("Pi:\n",np.around(p_i,3))
        # print("Fi:\n",np.around(f_i,3))
        new_links = np.random.uniform(size=i)
        new_links = np.where(new_links < p_i[:i], 1.0, 0.0)
        new_links = new_links * Epsilon_i * f_i[:i]
        new_links = np.concatenate([new_links,np.zeros(N-i)])
        # print("New Links:\n",np.around(new_links,2))
        new_am[i] += new_links
        
      #print(new_am.todense())

      new_am += new_am.transpose()
      
      # Remove alpha proportion of old edges
      remove_mask = np.where(np.random.uniform(size=Deg) < alpha, 0, 1)
      links.data = links.data*remove_mask
   
      new_am += in_am.multiply(links)
      
      new_am = new_am.tocsr()
      
      if not self.weighted:
        new_am.data = new_am.data.round()
        new_am.data = np.where(new_am.data > 0, 1.0, 0.0)
      elif self.discrete:
        new_am.data = new_am.data.round()
      
      return new_am

    
    # *****************************************
    # Back to main function, wtd_config_graph()
    # *****************************************
    
    pos_am = get_csr_pos(self.adj_mat)
    neg_am = get_csr_neg(self.adj_mat)
    
    total_am = csr_matrix(self.adj_mat.shape)
    
    if pos_am.getnnz() > 0:
      total_am += wtd_config_helper(pos_am, alpha=alpha)
    if neg_am.getnnz() > 0:
      total_am -= wtd_config_helper(neg_am, alpha=alpha)
    
    if ret_Exp_Graph:
      new_Exp_G = Experiment_Graph(type=self.type,am=total_am)
      return new_Exp_G
      
    return total_am
  

  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "c" - Real number [0,1], uniform infection probability (needed even if using 
  #               edge weighted diffusion, where it is ignored
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  #     "inoculated" - 1D np.array, length N, where 1 -> node @ index is inoculated (for epidemic mitigation)
  #                         inoculated nodes will never be infected unless they are also a seed-set node.
  # OUT: num_infected - natural number of infected nodes at end of Independent Cascade process
  #      infected - 1D np.array, length N, in {0,1}.  1 -> node @ index infected during IC process.
  # NOTE: 1) adjacency matrix must be symmetric
  #       2) does not currently support directed diffusion (note: need a multiplication on
  #           upper triangular and lower triangular adj matrices to make that work)
  def IC(self,**kwargs):
    
    try:
      seed_set = kwargs["seed_set"]
      infection_probability = kwargs["c"]
    except Exception as e:
      print("Issue with parameters in IC call",kwargs)
      print(e)
      return 0, set()
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    inoculated = kwargs.get("inoculated",None)
    
    active = seed_set.copy()
    recovered = np.ones(len(seed_set)) # This will be a mask we can element wise multiply to retrieve which 
    recovered -= active # nodes are susceptible. 1 <-> node susceptible, 0 o.w.'
    
    # set inoculated nodes immediately to recovered so they cannot participated in the process
    if isinstance(inoculated,np.ndarray):
      inoculated = np.where(active == 1, 0, inoculated).astype(int)
      recovered -= inoculated
      
    if debug:
      print("IC call on graph:",self.gname,":",self.nxgraph.order(),self.nxgraph.size())
      print("Params: c:,",infection_probability,"Wtd:",weighted)
      print("Total seeds:",np.sum(seed_set),"first 10 seeds:",np.argwhere(seed_set == 1).T[0][0:10])

    while np.sum(active) > 0:
      
      # Activations propagate along edges, so multiplying adj_mat X active yields a 1D array of new possible activations
      active_links = self.adj_mat.multiply(active)
      active_links.eliminate_zeros()
      active_links = csr_matrix(active_links.transpose().multiply(recovered).transpose())
      activation_sample = np.random.uniform(size=len(active_links.data))
      
      if weighted:
        active_links.data = np.where(activation_sample < active_links.data, 1.0, 0.0)
        active_links.eliminate_zeros()
      else:
        active_links.data *= np.where(activation_sample < infection_probability, 1.0, 0.0)
        active_links.eliminate_zeros()
             
      influence = active_links*active
      active = np.where(influence > 0, 1, 0)
      
      if isinstance(inoculated,np.ndarray):
        active = np.where(inoculated == 1, 0, active)
        
      if debug:
        print(np.argwhere(active == 1).T[0])
        
      recovered -= active
      
    infected = np.where(recovered == 1, 0, 1)
    if isinstance(inoculated,np.ndarray):
      infected = np.where(inoculated == 1, 0, infected)
      return self.N - (np.sum(infected) - np.sum(seed_set)) - np.sum(inoculated), infected
      
    num_infected = np.sum(infected) - np.sum(seed_set)
    return num_infected, infected
    
  # Alternate IC implementation which uses sets and the networkx graph instead
  # of the adjacency matrix to simulate process spread. Faster if the expected
  # number of nodes infected each timestep is very small and the graph is large.
  # Shares the same parameters as IC(), but can handle directed graphs. 
  # Unfortunately, self.nxgraph is not a directed graph by default, so to use you 
  # will have to make your own adjustments (or just extract this function). 
  def nx_IC(self,**kwargs):
  
    try:
      seed_set = kwargs["seed_set"]
      infection_probability = kwargs["c"]

    except Exception as e:
      print("Issue with parameters in nx_IC call",kwargs)
      print(e)
      return 0, set()

    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    directed = kwargs.get("directed",False)
    
    infected = set(np.argwhere(seed_set==1).T[0])
    volatile = set(np.argwhere(seed_set==1).T[0])

    while len(volatile) > 0:
      v_copy = volatile.copy()
      for v in v_copy:
        if directed:
          neighbors = set(self.nxgraph.successors(v))
        else:
          neighbors = set(self.nxgraph.neighbors(v))
          
        neighbors.difference_update(infected)
        
        for n in neighbors:
          if weighted:
            if r.random() < self.nxgraph[v][n]['weight']:
              infected.add(n)
              volatile.add(n)
          else:
            if r.random() < infection_probability:
              infected.add(n)
              volatile.add(n)
        volatile.remove(v)
        
      if debug:
        print(volatile)

    return len(infected)-np.sum(seed_set),infected
  
  
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "type" - "relative" : threshold is considered a relative fraction of infected neighbors
  #              "absolute" : threshold is considered an absolute weight or number of infected neighbors
  #              else : threshold is the minimum of the relative and absolute thresholds provided (per node)
  #     "r_threshold" : real number in [0,1], mandatory if "type" == "relative", activation threshold
  #     "a_threshold" : real number >= 0, mandatory if "type" == "absolute", activation threshold
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  # OUT: num_infected - natural number of infected nodes at end of the Linear Threshold (fixed weight) process
  #      infected - 1D np.array, length N, in {0,1}.  1 -> node @ index infected during LT process
  # NOTE: 1) adjacency matrix must be symmetric
  #       2) does not currently support directed diffusion (note: need a multiplication on
  #           upper triangular and lower triangular adj matrices to make that work)
  def LT(self,**kwargs):
    
    try:
      seed_set = kwargs["seed_set"]
      type = kwargs.get("type",'mixed')
      if type == 'relative':
        r_threshold = kwargs["r_threshold"]
        a_threshold = -1
      elif type == 'absolute':
        a_threshold = kwargs["a_threshold"]
        r_threshold = -1
      else:
        r_threshold = kwargs["r_threshold"]
        a_threshold = kwargs["a_threshold"]

    except Exception as e:
      print("Issue with parameters in LT call",kwargs)
      print(e)
      return 0, set()
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    
    active = seed_set.copy()
    recovered = np.ones(len(seed_set)) # This will be a mask we can element wise multiply to retrieve which 
    recovered -= active # nodes are susceptible. 1 <-> node susceptible, 0 o.w.
    
    if type == 'relative':
      in_edges = np.array(self.adj_mat.sum(axis=0)).reshape(self.adj_mat.shape[0])
      thresholds = in_edges*r_threshold
    elif type == 'absolute':
      thresholds = np.full(len(active),a_threshold)
    else:
      in_edges = np.array(self.adj_mat.sum(axis=0)).reshape(self.adj_mat.shape[0])
      r_thresholds = in_edges*r_threshold
      a_thresholds = np.full(len(active),a_threshold)
      thresholds = np.where(r_thresholds < a_thresholds, r_thresholds, a_threshold)
      
    thresholds = np.where(thresholds <= 0, 1, thresholds)
    influence = np.copy(active)*thresholds
    
    if debug:
      print("LT call on graph:",self.gname,":",self.nxgraph.order(),self.nxgraph.size())
      print("Params: Abs threshold:,",a_threshold,"Rel threshold:",r_threshold,"Type:",type,"Wtd:",weighted)
      print("Total seeds:",np.sum(seed_set),"first 10 seeds:",np.argwhere(seed_set == 1).T[0][0:10])

    while np.sum(active) > 0:
      
      active_links = self.adj_mat.multiply(active)
      active_links.eliminate_zeros()
      active_links = csr_matrix(active_links.transpose().multiply(recovered).transpose())
             
      influence += active_links*active
      susceptible_influence = influence*recovered

      active = np.where(susceptible_influence >= thresholds, 1, 0)

      if debug:
        print(influence)
        print(susceptible_influence)
        print(np.argwhere(active == 1).T[0])
        
      recovered -= active
      recovered = np.where(recovered < 0, 0, recovered)
      
    infected = np.where(recovered == 1, 0, 1)
    num_infected = np.sum(infected) - np.sum(seed_set)
    
    return num_infected, infected
  
  
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "type" - "relative" : threshold is considered a relative fraction of infected neighbors
  #              "absolute" : threshold is considered an absolute weight or number of infected neighbors
  #     "t1" : real number in [0,1] if relative or >= 0 if absolute; activation threshold
  #     "t2" : real number in [0,1] if relative or >= 0 if absolute; "saturation" threshold
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  # OUT: num_infected - natural number of infected nodes at end of the Saturated Linear Threshold process
  #      infected - 1D np.array, length N, in {0,1}.  1 -> node @ index infected during S-LT process
  # NOTE: 1) adjacency matrix must be symmetric
  #       2) does not currently support directed diffusion (note: need a multiplication on
  #           upper triangular and lower triangular adj matrices to make that work)
  def S_LT(self,**kwargs):
    
    try:
      seed_set = kwargs["seed_set"]
      type = kwargs.get("type",'relative')
      t1 = kwargs["t1"]
      t2 = kwargs["t2"]

    except Exception as e:
      print("Issue with parameters in LT call",kwargs)
      print(e)
      return 0, set()
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    
    active = seed_set.copy()
    recovered = np.ones(len(seed_set)) # This will be a mask we can element wise multiply to retrieve which 
    recovered -= active # nodes are susceptible. 1 <-> node susceptible, 0 o.w.
    
    saturated = np.zeros(len(seed_set))
    infected = np.copy(active)
    
    if type == 'relative':
      in_edges = np.array(self.adj_mat.sum(axis=0)).reshape(self.adj_mat.shape[0])
      active_thresholds = in_edges*t1
      saturate_thresholds = in_edges*t2
    elif type == 'absolute':
      active_thresholds = np.full(len(active),t1)
      saturate_thresholds = np.full(len(active),t2)

    active_thresholds = np.where(active_thresholds <= 0, 1, active_thresholds)
    saturate_thresholds = np.where(saturate_thresholds <= 0, 1, saturate_thresholds)
    
    if debug:
      print("LT call on graph:",self.gname,":",self.nxgraph.order(),self.nxgraph.size())
      print("Params: Type:,",type,"Active Threshold:",t1,"Saturation Threshold:",t2,"Wtd:",weighted)
      print("Total seeds:",np.sum(seed_set),"first 10 seeds:",np.argwhere(seed_set == 1).T[0][0:10])

    delta = True
    
    while delta:
      
      start_active = np.copy(active)
      
      active_links = self.adj_mat.multiply(active)
      active_links.eliminate_zeros()
      active_links = csr_matrix(active_links.transpose().multiply(recovered).transpose())
             
      influence = active_links*active
      
      saturated = np.where(influence >= saturate_thresholds, 1, saturated)
     
      active = np.where(influence >= active_thresholds, 1, active)
      active = np.where(saturated == 1, 0, active)
      infected = np.where(active == 1, 1, infected)

      if debug:
        print(np.round(influence[0:15],1))
        print(list(np.argwhere(active==1).T[0]))
        print(list(np.argwhere(saturated==1).T[0]))
        
      delta = not np.all(active == start_active)
        
    num_infected = np.sum(infected) - np.sum(seed_set)
    
    if debug:
      print()
      
    return num_infected, infected 
    
    
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "adv_set" - 1D np.array, length N, where 1 -> node @ index is an adversarial seed
  #     "type" - "relative" : threshold is considered a relative fraction of infected neighbors
  #              "absolute" : threshold is considered an absolute weight or number of infected neighbors
  #              "mixed"    : threshold is the minimum of the relative and absolute thresholds provided (per node)
  #     "r_threshold" : real number in [0,1], mandatory if "type" == "relative", activation threshold
  #     "a_threshold" : real number >= 0, mandatory if "type" == "absolute", activation threshold
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  # OUT: num_infected - natural number of infected nodes at end of the Adversarial Linear Threshold process
  #      infected - 1D np.array, length N, in {0,1}.  1 -> node @ index infected during ADV-LT process
  # NOTE: 1) adjacency matrix must be symmetric
  #       2) does not currently support directed diffusion (note: need a multiplication on
  #           upper triangular and lower triangular adj matrices to make that work)
  def ADV_LT(self,**kwargs):
    
    try:
      seed_set = kwargs["seed_set"]
      adv_set = kwargs["adv_set"]
      type = kwargs.get("type",'mixed')
      if type == 'relative':
        r_threshold = kwargs["r_threshold"]
        a_threshold = -1
      elif type == 'absolute':
        a_threshold = kwargs["a_threshold"]
        r_threshold = -1
      else:
        r_threshold = kwargs["r_threshold"]
        a_threshold = kwargs["a_threshold"]

    except Exception as e:
      print("Issue with parameters in LT call",kwargs)
      print(e)
      return 0, set()
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    
    active = seed_set.copy()
    
    # Keeps track of two interdependent processes in active and adv_active
    # susceptible and recovered nodes are shared among the two processes
    adv_active = adv_set.copy()
    contested = np.where(active + adv_active == 2, 1, 0)
    active -= contested
    adv_active -= contested
    recovered = np.ones(len(seed_set)) # This will be a mask we can element wise multiply to retrieve which 
    recovered -= active # nodes are susceptible. 1 <-> node susceptible, 0 o.w.
    recovered -= adv_active
    
    if type == 'relative':
      in_edges = np.array(self.adj_mat.sum(axis=0)).reshape(self.adj_mat.shape[0])
      thresholds = in_edges*r_threshold
    elif type == 'absolute':
      thresholds = np.full(len(active),a_threshold)
    else:
      in_edges = np.array(self.adj_mat.sum(axis=0)).reshape(self.adj_mat.shape[0])
      r_thresholds = in_edges*r_threshold
      a_thresholds = np.full(len(active),a_threshold)
      thresholds = np.where(r_thresholds < a_thresholds, r_thresholds, a_threshold)
      
    thresholds = np.where(thresholds <= 0, 1, thresholds)
    influence = active*thresholds
    adv_influence = adv_active*thresholds
    
    if debug:
      print("LT call on graph:",self.gname,":",self.nxgraph.order(),self.nxgraph.size())
      print("Params: Abs threshold:,",a_threshold,"Rel threshold:",r_threshold,"Type:",type,"Wtd:",weighted)
      print("Total seeds:",np.sum(seed_set),"first 10 seeds:",np.argwhere(seed_set == 1).T[0][0:10])

    num_infected = 0
    adv_num_infected = 0

    while np.sum(active) > 0:
      
      active_links = self.adj_mat.multiply(active)
      active_links.eliminate_zeros()
      active_links = csr_matrix(active_links.transpose().multiply(recovered).transpose())
             
      influence += active_links*active
      susceptible_influence = influence*recovered
      
      adv_active_links = self.adj_mat.multiply(adv_active)
      adv_active_links.eliminate_zeros()
      adv_active_links = csr_matrix(adv_active_links.transpose().multiply(recovered).transpose())
             
      adv_influence += adv_active_links*adv_active
      adv_susceptible_influence = adv_influence*recovered

      active = np.where(susceptible_influence >= thresholds, 1, 0)
      adv_active = np.where(adv_susceptible_influence >= thresholds, 1, 0)
      contested = np.where(active + adv_active == 2, 1, 0)
      
      active -= contested
      adv_active -= contested
      
      num_infected += np.sum(active)
      adv_num_infected += np.sum(adv_active)

      if debug:
        print(influence)
        print(susceptible_influence)
        print(np.argwhere(active == 1).T[0])
        
      recovered -= active
      recovered -= adv_active
      recovered -= contested
      recovered = np.where(recovered < 0, 0, recovered)
      
    infected = np.where(recovered == 1, 0, 1)
    
    return num_infected, adv_num_infected
    
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "c" - Real number [0,1], uniform infection probability (needed even if using 
  #               edge weighted diffusion, where it is ignored
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  #     "directed" - Boolean, allows for directed propagation
  #     "max_its" - Natural number, maximum time-steps to run the process before returning 
  #                   (if 0 process will run until termination)
  # OUT: num_infected - natural number of infected nodes at end of Ugander Complex process
  #      infected - set, members in {0,N-1}, all infected during UC process.
  # NOTE: infection factor is hard coded for all experiments to our estimate from Ugander et al's 
  #         "Structural diversity in social contagion" paper, but can be set as desired to tune behavior.
  def UC(self,**kwargs):
        
    try:
      seed_set = kwargs["seed_set"]
      c = kwargs["c"]

    except Exception as e:
      print("Issue with parameters in UC call",kwargs)
      print(e)
      return 0, set()
      
    ngraph = self.nxgraph
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    directed = kwargs.get("directed",False)
    T = kwargs.get("max_its",0)
    
    infection_factor = [0,0,1,1.5,.1]
    last_inff = len(infection_factor)-1
    
    # This updates the number of infected, connected-components in the 
    # neighborhood of node q.  
    def update_inf_comps(q,i_nbrs):
      for n in i_nbrs:
        curr_set = {n}
        n_nbrs = set_nbrs[n]
        i_comps = inf_comps[q].copy()
        for comp in i_comps:
          if len(n_nbrs.intersection(comp)) > 0:
            curr_set.update(comp)
            inf_comps[q].remove(comp)
        inf_comps[q].append(curr_set)
      return len(inf_comps[q])

    ## MAIN FUNCTION START ##
    
    set_nbrs = []
    for i in range(self.N):
      set_nbrs.append(set(ngraph.neighbors(i)))
    
    active = np.copy(seed_set)
    influenced = np.where(self.adj_mat * active > 0, 1, 0)
    this_susceptible = set(np.argwhere(influenced==1).T[0])
    
    init_State = set(np.argwhere(seed_set==1).T[0])
    infected = init_State.copy()
    delta = True
    count = 0
    
    all = set(ngraph.nodes)
    susceptible = all.difference(infected)
    this_susceptible.difference_update(infected)
    
    volatile = infected.copy()  # holds last round's additions to infected
                                # which are the only nodes we need to add to inf_comps
    
    distinct_infected = [0]*self.N
    inf_comps = [[] for i in range(self.N)] 

    while delta:
      if count >= T and T > 0:
        return len(infected)-len(init_State),infected
      count += 1

      delta = False
      next_infected = set()
      next_susceptible = set()
      
      for i in this_susceptible:
        neighbors = set_nbrs[i].copy()
        neighbors.intersection_update(volatile)
        
        ccs = update_inf_comps(i,neighbors)
        
        # For this process, the first time a node is exposed to exactly k 
        # infected, connected-components in its neighborhood, with probability
        # infection_factor[k] that node transitions to the infected state
        for event in range(distinct_infected[i],ccs):
          inf_fact = infection_factor[min(event+1,last_inff)]*c
          if r.random() < inf_fact:
            next_infected.add(i)
            next_susceptible.update(set_nbrs[i])
            delta = True
            continue
            
        distinct_infected[i] = max(ccs,distinct_infected[i])
        
      infected.update(next_infected)
      susceptible.difference_update(next_infected)
      volatile = next_infected.copy()
      this_susceptible = next_susceptible.intersection(susceptible)

    num_infected = len(infected) - len(init_State)
    return num_infected,infected
    
    
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "c" - Real number [0,1], uniform infection probability (needed even if using 
  #               edge weighted diffusion, where it is ignored
  #     "adv_set" - 1D np.array, length N, where 1 -> node @ index is an adversarial seed
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  #     "directed" - Boolean, allows for directed propagation
  #     "max_its" - Natural number, maximum time-steps to run the process before returning 
  #                   (if 0 process will run until termination)
  # OUT: num_infected - natural number of infected nodes at end of the Adversarial Ugander Complex process
  #      infected - set, members in {0,N-1}, all infected during ADV-UC process.
  # NOTE: infection factor is hard coded for all experiments to our estimate from Ugander et al's 
  #         "Structural diversity in social contagion" paper, but can be set as desired to tune behavior.
  def ADV_UC(self,**kwargs):
        
    try:
      seed_set = kwargs["seed_set"]
      c = kwargs["c"]
      adv_set = kwargs["adv_set"]

    except Exception as e:
      print("Issue with parameters in ADV_UC call",kwargs)
      print(e)
      return 0, set()
      
    ngraph = self.nxgraph
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    directed = kwargs.get("directed",False)
    T = kwargs.get("max_its",0)

    infection_factor = [0,0,1,1.5,.1]
    last_inff = len(infection_factor)-1
    
    # This updates the number of infected, connected-components in the 
    # neighborhood of node q.  
    def update_inf_comps(q,i_nbrs,adv=False):
      if adv:
        q_inf_comps = adv_inf_comps[q]
      else:
        q_inf_comps = inf_comps[q]
        
      for n in i_nbrs:
        curr_set = {n}
        n_nbrs = set_nbrs[n]
        i_comps = q_inf_comps.copy()
        for comp in i_comps:
          if len(n_nbrs.intersection(comp)) > 0:
            curr_set.update(comp)
            q_inf_comps.remove(comp)
        q_inf_comps.append(curr_set)
      return len(q_inf_comps)

    ## MAIN FUNCTION START ##
    
    set_nbrs = []
    for i in range(self.N):
      set_nbrs.append(set(ngraph.neighbors(i)))
    
    active = np.copy(seed_set)
    adv_active = np.copy(adv_set)
    influenced = np.where(self.adj_mat * active > 0, 1, 0)
    influenced += np.where(self.adj_mat * adv_active > 0, 1, 0)
    this_susceptible = set(np.argwhere(influenced>0).T[0])
    
    init_State = set(np.argwhere(seed_set==1).T[0])
    infected = init_State.copy()
    adv_init = set(np.argwhere(adv_set==1).T[0])
    adv_infected = adv_init.copy()
    
    delta = True
    count = 0
    
    # Keep track of two interdependent processes in active and adv_active
    # susceptible and recovered nodes are shared among the two processes
    all = set(ngraph.nodes)
    susceptible = all.difference(infected)
    susceptible = all.difference(adv_infected)
    
    this_susceptible.difference_update(infected)
    this_susceptible.difference_update(adv_infected)
    
    volatile = infected.copy()  # holds last round's additions to infected
                                # which are the only nodes we need to add to inf_comps
    adv_volatile = adv_infected.copy()
    
    distinct_infected = [0]*self.N
    inf_comps = [[] for i in range(self.N)] 
    
    adv_distinct_infected = [0]*self.N
    adv_inf_comps = [[] for i in range(self.N)] 

    while delta:
      if count >= T and T > 0:
        return len(infected)-len(init_State),len(adv_infected)-len(adv_init)
      count += 1

      delta = False
      next_infected = set()
      next_susceptible = set()
      
      adv_next_infected = set()
      adv_next_susceptible = set()
      #print(len(susceptible),len(this_susceptible),len(adv_infected),len(infected))
      
      for i in this_susceptible:
        neighbors = set_nbrs[i].copy()
        adv_neighbors = neighbors.intersection(adv_volatile)
        neighbors.intersection_update(volatile)
        
        ccs = update_inf_comps(i,neighbors)
        ccs = len(inf_comps[i])
        
        adv_ccs = update_inf_comps(i,adv_neighbors,adv=True)
        adv_ccs = len(adv_inf_comps[i])

        # For this process, the first time a node is exposed to exactly k 
        # infected, connected-components in its neighborhood, with probability
        # infection_factor[k] that node transitions to the infected state
        for event in range(distinct_infected[i],ccs):
          inf_fact = infection_factor[min(event+1,last_inff)]*c
          if r.random() < inf_fact:
            next_infected.add(i)
            delta = True
            continue
            
        for event in range(adv_distinct_infected[i],adv_ccs):
          inf_fact = infection_factor[min(event+1,last_inff)]*c
          if r.random() < inf_fact:
            adv_next_infected.add(i)
            delta = True
            continue  
            
        distinct_infected[i] = max(ccs,distinct_infected[i])
        adv_distinct_infected[i] = max(ccs,adv_distinct_infected[i])
        
      contested = adv_next_infected.intersection(next_infected)
      
      all_next = adv_next_infected.union(next_infected)
      all_next.difference_update(contested)
      susceptible.difference_update(contested)
      
      for next_infectee in all_next:
        next_susceptible.update(set_nbrs[next_infectee])
        
      infected.update(next_infected)
      adv_infected.update(adv_next_infected)
      
      volatile = next_infected.copy()
      adv_volatile = adv_next_infected.copy()

      susceptible.difference_update(next_infected)
      susceptible.difference_update(adv_next_infected)
      this_susceptible = next_susceptible.intersection(susceptible)

    num_infected = len(infected) - len(init_State)
    adv_num_infected = len(adv_infected) - len(adv_init)
    
    return num_infected, adv_num_infected    
    
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "c" - Real number [0,1], uniform infection probability (needed even if using 
  #               edge weighted diffusion, where it is ignored
  #   Optional:
  #     "weighted" - Boolean, if true then edge weights must be in [0,1] and represent probability
  #                     of transmission along that edge
  #     "debug" - Boolean, offers some debug statements to track progress of diffusion
  #     "mode" - 0: Proportion of infected neighbors if each neighbor can only represent one community, 
  #                   as compared to the maximum possible for the node (if all neighbors were infected).
  #              1: Minimum of : (neighbors infected / total neighbors) AND /# of communities infected  / total communities    \
  #                                                                         \    neighbors represent   /  of all neighbors     /
  #              2: Average of the two quantities in mode 1
  #              else: fraction of node i's communities for which i has a neighbor who has adopted
  #     "max_its" - Natural number, maximum time-steps to run the process before returning 
  #                   (if 0 process will run until termination)
  # OUT: num_infected - natural number of infected nodes at end of Ugander Complex process
  #      infected - set, members in {0,N-1}, all infected during UC process.
  # NOTE: 1) Assumes a nxgraph which has nodes with attribute 'num_comms' = number of communities it belongs to
  #          and edges which have a 'community' attribute which specifies which communit(y/ies) connect(s) the two nodes
  #       2) Mode will greatly effect the speed of the process. If mode is 0 the process will run slowly, as calculating the
  #          "one vote each" representation is non-trivial. We recommend using one of the other modes (particularly 1) for 
  #          most practical cases as we expect the behavior to be similar.
  def CC(self,**kwargs):
        
    try:
      seed_set = kwargs["seed_set"]
      threshold = kwargs["threshold"]
      c = kwargs["c"]

    except Exception as e:
      print("Issue with parameters in UC call",kwargs)
      print(e)
      return 0, set()
      
    ngraph = self.nxgraph
      
    weighted = kwargs.get("weighted",False)
    debug = kwargs.get("debug",False)
    mode = kwargs.get("mode",0)
    T = kwargs.get("max_its",0)

    directed = False
    delta = True
    ORF_dict = {}

    # This calculates the highest possible number of distinct 
    # communities which a set of neighbors could represent if
    # each member could only represent one of its communities
    def one_rep_frac(neighbors):

      comm_inf = set()
      n2c = []
      c2n = []
      for i_nbr in neighbors:
        n_comms = ngraph[i][i_nbr]['communities']
        known_thru = []
        for n in n_comms:
          known_thru += [e for e in ngraph.nodes[i_nbr]['communities'] if n in e]
        n2c.append(known_thru)
        comm_inf.update(known_thru)
        
      ci = 0
      for c in comm_inf:
        ni = 0
        c2n.append([])
        for n_comms in n2c:
          if c in n_comms:
            c2n[ci].append(ni)
          ni += 1
        ci += 1
      
      comm_rep = set()
      while max([len(e) for e in n2c]) > 0 and max([len(e) for e in c2n]) > 0:
        least = min([len(e) for e in c2n if len(e) > 0])
        LRC = [i for (i,j) in enumerate(c2n) if len(j)==least][0]
        smallest = min([len(n2c[e]) for e in c2n[LRC]])
        LCN = [i for (i,j) in enumerate(n2c) if len(j)==smallest][0]
        p_removed = n2c[LCN]
        n2c[LCN] = []
        c2n[LRC] = []
        [i.remove(LRC) for i in n2c if LRC in i]
        [i.remove(LCN) for i in c2n if LCN in i]
        comm_rep.add(LRC)   
      
      return(len(comm_rep))
        
    def remember_ORF_total(node):
      if ORF_dict.get(node) == None:
        neighbors = set_nbrs[node]
        ORF_dict[node] = one_rep_frac(neighbors)
      return ORF_dict[node]
      
    set_nbrs = []
    for i in range(self.N):
      set_nbrs.append(set(ngraph.neighbors(i)))
    
    active = np.copy(seed_set)
    influenced = np.where(self.adj_mat * active > 0, 1, 0)
    this_susceptible = set(np.argwhere(influenced==1).T[0])
   
    init_State = set(np.argwhere(seed_set==1).T[0])
    infected = init_State.copy()
    delta = True
    count = 0
    
    all = set(ngraph.nodes)
    susceptible = all.difference(infected)
    this_susceptible.difference_update(infected)    

    while delta:
    
      if debug:
        print("A:",len(this_susceptible),"I:",len(infected),"S:",len(susceptible))
        
      delta = False
      next_infected = set()
      next_susceptible = set()
      
      for i in this_susceptible:

        neighbors = set_nbrs[i].copy()
        t_nbrs = len(neighbors)

        neighbors.intersection_update(infected)
        
        if len(neighbors) == 0:
          continue
          
        comm_inf = set()
        for i_nbr in neighbors:
          comm_inf.update(ngraph[i][i_nbr]['communities'])
          
        nbrfrac = len(neighbors) / t_nbrs
        cfrac = len(comm_inf) / len(ngraph.nodes[i]['comm_acq'])
        
        if mode == 0:
          # FRAC INF / MAX POSS 1 REP
          total_ORF = remember_ORF_total(i)
          if debug:
            print(total_ORF,":",one_rep_frac(neighbors),end="\t")
          i_frac = one_rep_frac(neighbors) / total_ORF
        elif mode == 1:
          # MIN(Frac Nbrs infected, Frac Comm Acq Inf)
          i_frac = min(nbrfrac,cfrac)
        elif mode == 2:
          # AVG(Frac Nbrs infected, Frac Comm Acq Inf)
          i_frac = (nbrfrac + cfrac)/2
        else:
          # fraction of i's communities for which i has a neighbor who has adopted
          i_frac = len(comm_inf) / len(ngraph.nodes[i]['comm_acq'])                                                         

        if i_frac >= threshold and r.random() < c:
          delta = True
          next_infected.add(i)
          next_susceptible.update(set_nbrs[i])
      
      # Update infected, susceptible, and next_susceptible 
      infected.update(next_infected)
      susceptible.difference_update(next_infected)
      this_susceptible = next_susceptible.intersection(susceptible)
    
    return len(infected)-len(init_State),infected
     

  # IN: ks - list(integers), desired seed-set sizes
  #     L - real number > 0, determines probability (=1-N^(-L)) that seed sets returned are
  #          outside the approximation guarantee. 1 is a good setting for moderate or larger graphs.
  #     eps - (small) real number in (0,1], determines strength of approximation guarantee and
  #           is the greatest determinant of the speed of the algorithm (factor of 1/e^2)
  #     c - real number in [0,1], uniform infection probability. If left at -1, infection probabilities
  #           will be taken as edge weights.
  # OUT: seed_sets - list(lists(integers)), greedy seed-set's to maximize Independent Cascade influence 
  #          using Algorithm 2 from Tang's "Influence Maximization: Near-Optimal Time Complexity Meets Practical Efficiency"
  #      theta - integer > 0, number of reverse-reachable sets generated to conduct the algorithm
  # NOTE: This is not the fastest algorithm in Tang's paper, but is the most straightforward implementation of their
  #       primary technique. Users wishing the most optimized algorithm should use Alg 3 from their paper.
  def Tang_Seed(self,ks,L,eps,c=-1):
  
    n = self.N
    m = self.M
    ks.sort()
    k = ks[-1]
    ngraph = self.nxgraph
    
    nbrs = []
    for i in range(n):
      nbrs.append(set(ngraph.neighbors(i)))
      
  
    def Tang_KPT_est():
      
      log_n = np.log2(n)
      ln_n = np.log(n)
      
      for i in range(1,int(log_n)):
        
        c_i = (6*L*ln_n+6*np.log(log_n))*2**i
        sum = 0
        avg_width = 0
        for j in range(1,int(c_i)+1):
          rr_set = RR()
          avg_width += width(rr_set)
          K_R = 1 - (1 - ( width(rr_set) / m ))**k
          #print(K_R,width(rr_set),m,width(rr_set) / m, (1-width(rr_set) / m),(1-width(rr_set) / m)**k)
          sum += K_R
        if sum/c_i > (.5**i):
          return ( n*sum / (2*c_i) )
      return 1
      
    def comp_theta():
      
      logn = np.log(n)
      nCk = spesci.comb(n,k)
      lam = (8+2*eps)*n*(L*logn+np.log(nCk)+np.log(2))*eps**(-2)
      theta = int(lam / KPT)
      
      return theta
      
    def width(rr_set):
      
      width = 0
      for node in rr_set:
        width += ngraph.degree(node)
        
      return width/2
              
    def RR():
      
      v = r.sample(range(n),1)
      
      infected = set()
      volatile = set()
      
      volatile.add(v[0])
      infected.add(v[0])

      while len(volatile) > 0:
        v_copy = volatile.copy()
        for v in v_copy:
          # if directed:
            # neighbors = set(ngraph.predecessors(v))
          # else:
          neighbors = nbrs[v].copy()
            
          neighbors.difference_update(infected)
          
          for nbr in neighbors:
            if c == -1:
              if r.random() < ngraph[nbr][v]['weight']:
                infected.add(nbr)
                volatile.add(nbr)
            else:
              if r.random() < c:
                infected.add(nbr)
                volatile.add(nbr)
          volatile.remove(v)
        
      return infected
     
    KPT = Tang_KPT_est()
    theta = comp_theta()

    r_sets = []
    seeds = []
    node_coverage = {}
    for node in ngraph.nodes():
      node_coverage[node] = 0
      
    for x in range(0,theta):  #generate theta RR sets and update coverage
      r_sets.append(RR())
      for node in r_sets[-1]:
        node_coverage[node] += 1
        
    for x in range(0,k):
      node = max(node_coverage, key=node_coverage.get)
      seeds.append(node)   #add node which covers most RR sets to seeds
      
      removed_sets = []  #keep track of coverage of nodes and add 
      for rr_set in r_sets:    #any RR which newest node covers to 
        if node in rr_set:  #removed_sets to be removed from r_sets
          for v in rr_set:
            node_coverage[v] -= 1
          removed_sets.append(rr_set)
          
      for rr_set in removed_sets:
        r_sets.remove(rr_set)
        
    
    seed_sets = [seeds[0:q] for q in ks]
    return seed_sets, theta
    
  # IN: (am) - np.array, if provided uses the given array instead of self.adj_mat
  # OUT: None
  # Prints some simple measures of a 2D array interpreted as an adjacency matrix.
  def analyze_am(self, am=np.array([])):
  
    if not np.any(am):
      am = self.adj_mat
      
    links = np.where(am != 0, 1.0, am)
    degrees = np.sum(links,axis=1)
    strengths = np.sum(am,axis=1)
    
    print("Edges",np.sum(degrees),"\tDeg seq:\n",degrees)
    print("Total Str",np.round(np.sum(strengths),2),"\tAvg Link Str",np.round(np.sum(strengths)/np.sum(degrees),2),"\tStr seg:\n",np.around(strengths,2))
    print("Links:\n",links)
    print("AM\n",np.around(am,2))

  # IN: structure_label - String, key for sructure to be stored
  #     structure - a structure of the graph (e.g. a partition from community detection)
  # OUT: None
  # self.structures updated
  def add_structure(self,structure_label,structure):
    self.structures[structure_label] = structure
    
  # IN: None
  # OUT: Returns most likely SBM recovered between the Classic or Degree-Corrected models
  def best_SBM(self):
    if "sbm" in self.structures:
      if "dc_sbm" in self.structures:
        if self.structures["dc_sbm"].entropy() < self.structures["sbm"].entropy():
          return self.structures["dc_sbm"],"dc_sbm"
      return self.structures["sbm"],"sbm"
    return None,"No SBM fit"
    
  # Convenience function for plotting robustness as it is stored by structure_test
  def plot_robustness(self,subplot=None):
  
    if subplot == None:
      fig, subplot = plt.subplots(1)
      
    colors = ["red","green","blue","orange","purple"]
    
    for i in range(len(self.results["SDAs"])):
      subplot.scatter(self.results["alphas"],self.results["VIC_Configs"][i],c=colors[i],label=self.results["SDAs"][i])
  
    plt.show()
    
  # IN: (path)
  # OUT: None
  # pickles the Experiment_Graph object at path (or self.path) 
  def pickle(self,path=""):
    try:
      if path == "":
        file = self.path + self.filename
      else:
        file = path + self.filename
      with open(file, 'wb+') as fp:
        pickle.dump(self, fp)
    except Exception as e:
      print("Issue saving",self.type,self.gname)
      print(e)
      
  # IN: structure_detect_alg - String {"sbm","dc_sbm","infomap"} determines which community detection alg to run
  #     (runs) - integer number of times to execute the detection algorithm
  #     (add) - Boolean, whether to save the partition in self.structures
  #     (reset) - Boolean, whether to ignore any previously saved structures
  # OUT: structure as returned according to the structure detection algorithm selected
  def get_structure(self,structure_detect_alg,runs=5,add=False,reset=False):
    
    if not reset and structure_detect_alg in self.structures:
      return self.structures[structure_detect_alg]
    
    min_Score = np.Inf
    current_best = None
    for x in range(runs):
      candidate_structure = self.structure_detect(structure_detect_alg) 
      candidate_strength = self.strength(candidate_structure,structure_detect_alg)
      if candidate_strength < min_Score:
        current_best = candidate_structure
        min_Score = candidate_strength
    if add:
      self.add_structure(structure_detect_alg,current_best)
    return current_best
    
  # IN: structure - object returned by infomap or SBM recovery
  #     SDA - String {"sbm","dc_sbm","infomap"}
  # OUT: 1D np.array, partition of nodes according to structure given
  def partition(self,structure,SDA):
    if SDA == "sbm" or SDA == "dc_sbm":
      return structure.get_blocks().a
    elif SDA == "infomap":
      return np.array(structure.membership)
    
  # IN: structure - object returned by infomap or SBM recovery
  #     SDA - String {"sbm","dc_sbm","infomap"}
  # OUT: real number, modularity of partition according to structure given
  def modularity(self,structure,SDA):
    if structure == None:
      structure = self.get_structure(SDA)

    if SDA == "sbm" or SDA == "dc_sbm":
      partition = structure.get_blocks()
      sda_partition = self.gtgraph.new_vertex_property("int32_t")
      self.gtgraph.vp.part = sda_partition
      self.gtgraph.vp.part.a = partition.a

      return gt.modularity(self.gtgraph,self.gtgraph.vp.part)#,weight=self.gtgraph.ep.weight)
    elif SDA == "infomap":
      return structure.q    
    
  # IN: structure - object returned by infomap or SBM recovery
  #     SDA - String {"sbm","dc_sbm","infomap"}
  # OUT: entropy or codelength according to structure given
  def strength(self,candidate,structure_detect_alg):
    if structure_detect_alg == "sbm":
      return candidate.entropy()
    elif structure_detect_alg == "dc_sbm":
      return candidate.entropy()
    elif structure_detect_alg == "infomap":
      return candidate.codelength
    
  # IN: structure_detect_alg - String {"sbm","dc_sbm","infomap"}
  # OUT: structure according to structure_detect_alg specified, using graph-tool SBM recovery or
  #      igraph's infomap algorithm
  # NOTE: this is a single recovery process, since the results are stochastic
  #   we recommend using get_structure instead and setting runs to an appropriate
  #   small integer.
  def structure_detect(self,structure_detect_alg):
    if structure_detect_alg == "sbm":
      g = self.gtgraph
      if self.weighted:
        return gt.minimize_blockmodel_dl(g,state_args=dict(recs=[g.ep.weight],rec_types=[self.rec_type]),deg_corr=False)
      return gt.minimize_blockmodel_dl(g,deg_corr=False)
    elif structure_detect_alg == "dc_sbm":
      g = self.gtgraph
      if self.weighted:
        return gt.minimize_blockmodel_dl(g,state_args=dict(recs=[g.ep.weight],rec_types=[self.rec_type]))
      return gt.minimize_blockmodel_dl(g,deg_corr=True)
    elif structure_detect_alg == "infomap":
      return self.igraph.community_infomap(edge_weights='weight',trials=1)
      
  # Convenience method to set a single graph layout for consistent drawing.
  def set_pos(self):
    self.pos = nx.spring_layout(self.nxgraph)
      
  # IN: partition - 1D np.array of node community affiliations
  #     (pos) - networkx graph layout for drawing
  #     (subplot) - matplotlib axis to draw in 
  # OUT: None
  # Draws the graph with nodes colored by the community affiliations in partition
  def draw_graph_communities(self,partition,pos=None,subplot=None):
  
    if subplot == None:
      fig, subplot = plt.subplots(1)
    if pos == None:
      pos = nx.spring_layout(self.nxgraph)
    
    plt.sca(subplot)
    
    drawn_graph = nx.Graph()

    cmap = "tab10"
    cmap = plt.get_cmap(cmap)
    num_comms = np.amax(partition) + 1
    colors = cmap(range(num_comms+1))
      
    graph = nx.Graph()
    
    for e in self.nxgraph.edges():
      icom = partition[e[0]]
      jcom = partition[e[1]]
      if icom != jcom:
        graph.add_edge(e[0],e[1],color=colors[num_comms])
      else:
        graph.add_edge(e[0],e[1],color=colors[icom])

    edge_colors = []
    for e in graph.edges(data='color'):
      edge_colors.append(e[2])
      
    node_labels = {node:node for node in graph.nodes}
    node_colors = [colors[partition[node]] for node in graph.nodes]
    
    nx.draw(graph,pos,edge_color=edge_colors,width=.2,node_size=15, node_color=node_colors)
    
# In: Two partitions, A, B, which are np.arrays of INT of equal length
# Out: V(A,B) = H(A) + H(B) - 2I(A,B), a number in 0-1
# Normalized Variation of Information
def VIC(A,B):
  return (H(A) + H(B) -2*mutual_info_score(A,B))/np.log(len(A))

# In: Two partitions, A, B, which are np.arrays of INT of equal length
# Out: V(A,B) = H(A) + H(B) - 2I(A,B), a number in 0-1
# Normalized Mutual Information
def I_norm(A,B):
  return 2*mutual_info_score(A,B)/(H(A)+H(B))
  
# In: Partition A, np.array of INT
# Out: H(A) = - SUM_a: P(A)*log[P(A)]
# Entropy of list/array A
def H(A):
  try:
    classes,counts = np.unique(A, return_counts=True)
    entropy = scipy.stats.entropy(counts)  # get entropy from counts
    return entropy
  except Exception as e:
    print(e)
    print("issue calculating entropy of:")
    print(A)
    return 0

# Returns a sparse matrix with only the negative elements in cm
def get_csr_neg(cm):
  neg = cm.copy()
  neg.data = np.where(neg.data < 0, -neg.data,0)
  res = csr_matrix(neg)
  res.eliminate_zeros()
  return res

# Returns a sparse matrix with only the positive elements in cm 
def get_csr_pos(cm):
  pos = cm.copy()
  pos.data = np.where(pos.data > 0, pos.data, 0)
  res = csr_matrix(pos)
  res.eliminate_zeros()
  return res

# IN: filename - String, filename of a .edges file to read from
# OUT: edges - 2-column np.array, each row is a node pair denoting edges in a graph
#      weights - 1D np.array(float or integer) denoting edge weights
# NOTE: 1) some regular expression work was done to make this work with a majority of 
#      .edges files from NetworkRepository.com. It should work on most lists of edges,
#      in particular any of the following forms, where the square brackets denote 
#      the format of each line, and the spaces can be any whitespace characters:
#      [int int], [int int float], [int int float ******]
#   2) filepath is currently hardcoded
def read_edges(graph_filename):

  fh=open("graphs//Network_Repository//"+graph_filename+".edges", encoding='utf-8')
  lines = [a.strip() for a in fh.readlines()]
  fh.close()
  
  edge_lines = [re.sub(r"[\s,]+", " ", a) for a in lines if a[0] != '%']
  edge_lines = [e.split() for e in edge_lines]
  edge_line_length = len(edge_lines[0])
  
  origin = [e[0] for e in edge_lines]
  dest = [e[1] for e in edge_lines]
  
  og_nodes = list(set(origin).union(set(dest)))
  og_indices = {og_nodes[i]:i for i in range(len(og_nodes))}
  
  
  origin = [og_indices[e] for e in origin]
  dest = [og_indices[e] for e in dest]
  
  edges = np.stack((np.array(origin), np.array(dest)))
  
  if edge_line_length >= 3:
    weights = np.array([float(e[2]) for e in edge_lines])
  else:
    weights = np.ones(len(edge_lines))
  
  return edges, weights 
  
# IN: graphlist_file - list(String), filenames of .edges files to read from
#     (reload) - Boolean, whether to read from .edges file or attempt to load a pickle first
# OUT: list(Experiment_Graph) - all experiment graphs successfully loaded from graphlist_file
# NOTE: 1) filepath is currently hardcoded
def read_graphs_from_list(graphlist_file,reload=False):

  path = "graphs//Network_Repository//"
  fh=open(path+graphlist_file+".txt", encoding='utf-8')
  lines = [a.strip() for a in fh.readlines()]
  fh.close()
  
  graph_list = []

  for line in lines:
    graph_name, graph_type = line.split()
    filename = graph_type + "//" + graph_name
    if ospath.exists(path+filename) and not reload:
      with open(path+filename,'rb') as f:
        graph = pickle.load(f)
      graph.gname = graph_name
      graph_list.append(graph)
    else:
      try:
        graph = Experiment_Graph(graph_type,read_data = read_edges(filename), filename=filename,path=path,gname=graph_name)
        graph_list.append(graph)
      except Exception as e:
        print("Issue with",graph_type,graph_name)
        print(e)
    
  return graph_list
  
# IN: graph_name - String, filename of a .edges file to read from
#     graph_type - String, category and subfolder on hardcoded path below to find .edges file
#     (reload) - Boolean, whether to read from .edges file or attempt to load a pickle first
# OUT: experiment graph - Experiment_Graph if successfully loaded from path+graph_type+graph_name
# NOTE: 1) filepath is currently hardcoded
def load_graph(graph_name,graph_type,reload=False):
  
  path = "graphs//Network_Repository//"
  filename = graph_type + "//" + graph_name
  if ospath.exists(path+filename) and not reload:
    with open(path+filename,'rb') as f:
      graph = pickle.load(f)
    graph.gname = graph_name
    return graph
  else:
    try:
      graph = Experiment_Graph(graph_type,read_data = read_edges(filename), filename=filename,path=path,gname=graph_name)
      return graph
    except Exception as e:
      print("Issue with",graph_type,graph_name)
      print(e)
      return None
  
# IN: graph - Experiment_Graph object to test robustness 
#     SDAs - list(strings), identifies which structure detection algorithms to use when testing robustness
#     (alphas) - list(floats [0,1]), identifies how much to perturb the graph during testing
#     (runs) - integer, number of times to run detection algorithm on perturbed graphs
#     (store) - Boolean, whether or not to save robustness results in graph.results
#     (reset_part) - Boolean, whether or not to reset previously saved partitions on graph
#     (status) - Boolean, toggles output of progress through test
# OUT: robustness - list(float) - 1-VIC for averaged over all alphas, for each SDA on graph
#      VIC_Original - list(list(float)) - VIC for all alpha, SDA pairs on graph
# NOTE: Please refer to Karrer, Levina, and Newman's work "Robustness of community structure in networks" 
#     and my thesis for the principles behind robustness detection. 
def structure_test(graph,SDAs,X=5,alphas=[i*.05 for i in range(8)],runs=5,store=True,reset_part=False,reset_robust=False,
  status=False):

  if not reset_robust and "robustness" in graph.results:
    return graph.results["robustness"], graph.results["VIC_Original"]
    
  alphas = [round(i,2) for i in alphas]

  VIC_Original = np.zeros((len(SDAs),len(alphas)))
  time_sdas = [0,0,0]
  partitions = {}
  
  t1 = 0
  for SDA in SDAs:
    if SDA == "skip1" or SDA == "skip2":
      continue
    if status:
      sys.stdout.write("\rSDA {}; On Original graph: ({},{}) last in {}s            "
       .format(SDA, graph.nxgraph.order(), graph.nxgraph.size(),t1))
      sys.stdout.flush()
    s = time.time()
    partitions[SDA] = graph.partition(graph.get_structure(SDA,runs=runs,reset=reset_part,add=True),SDA)
    e = time.time()
    t1 = np.round(e-s,2)
    
  graph.pickle()
  
  t1,t2 = [0,0]
  if status:
    print("\nBeginning Perturbations")
  for iteration in range(X):
    for alpha in alphas[1:]:
      # sys.stdout.write("\rIteration {}; On Alpha:{}/{}; Generating config graph ({},{})"
        # .format(iteration,alpha,len(alphas)-1,graph.nxgraph.order(),graph.nxgraph.size()))
      # sys.stdout.flush()
      graph_a = graph.wtd_config_graph(alpha=alpha,ret_Exp_Graph=True)
      if graph_a.gtgraph.num_vertices() != graph.gtgraph.num_vertices():
        print("Generated graph of unequal vertices:",graph.name,graph_a.gtgraph.num_vertices(),graph.num_vertices())
      
      for SDA in SDAs:
        if SDA == "skip1" or SDA == "skip2":
          VIC_Original[SDAs.index(SDA),alphas.index(alpha)] = -1.0
          continue
        if status:
          sys.stdout.write("\rIteration {}; On Alpha:{}/{}; SDA:{} on graph ({},{}) last in {}s,{}s                "
            .format(iteration,alpha,len(alphas)-1,SDA,graph.nxgraph.order(),graph.nxgraph.size(),t1,t2))
          sys.stdout.flush()
        s = time.time()
        partition = graph_a.partition(graph_a.get_structure(SDA,runs=1),SDA)
        e = time.time()
        t1 = np.round(e-s,1)
        s = time.time()
        VIC_Original[SDAs.index(SDA),alphas.index(alpha)] += VIC(partition,partitions[SDA])
        e = time.time()
        t2 = np.round(e-s,1)
        
        
  VIC_Original /= X
  if status:
    print()

  robustness = np.sum(VIC_Original,axis=1)
  
  if store:
    graph.results["VIC_Original"] = VIC_Original
    graph.results["robustness"] = robustness
    graph.results["alphas"] = alphas
    graph.results["SDAs"] = SDAs
    
  return robustness, VIC_Original
  
# This is a visualization method which displays the communities detected as the graph
# is perturbed according to alphas, using a single structure / community detection algorithm SDA. 
def inspect_robust(graph,SDA,alphas=[i*.05 for i in range(6)],runs=5):

  #VIC_Configs = np.zeros((len(alphas)))
  VIC_Original = np.zeros((len(alphas)))
  partitions = {}
  
  print(graph.nxgraph.order(),graph.nxgraph.size())
  pos = nx.spring_layout(graph.nxgraph)
  fig, plots = plt.subplots(1,len(alphas)+1)
  fig.tight_layout()
  
  # G_ = graph.wtd_config_graph(ret_Exp_Graph=True)
  # partitions[SDA] = G_.partition(G_.get_structure(SDA,runs=runs),SDA)
  # G_.draw_graph_communities(partitions[SDA],pos=pos,subplot=configs[0])

  # for alpha in alphas[1:]:
    # G_a = G_.wtd_config_graph(alpha=alpha,ret_Exp_Graph=True)
    # partition = G_a.partition(G_a.get_structure(SDA,runs=runs),SDA)
    
    # G_a.draw_graph_communities(partition,pos=pos,subplot=configs[alphas.index(alpha)])
    # VIC_Configs[alphas.index(alpha)] += VIC(partition,partitions[SDA])

  # print("\nConfig\n",partitions[SDA])
  
  partitions[SDA] = graph.partition(graph.get_structure(SDA,runs=runs,reset=True,add=True),SDA)
  graph.draw_graph_communities(partitions[SDA],pos=pos,subplot=plots[1])
  # print("\nOriginal\n",partitions[SDA])
  
  for alpha in alphas[1:]:
    graph_a = graph.wtd_config_graph(alpha=alpha,ret_Exp_Graph=True)
    
    ew = graph_a.gtgraph.ep.weight.a
    # print("Non integer edges for mutated graph:")
    # print(graph_a.weighted,graph_a.rec_type)
    # print(ew[(ew != 0.0) & (ew != 1.0)])

    partition = graph_a.partition(graph_a.get_structure(SDA,runs=runs),SDA)
    print("\nPerturbed:",alpha,"best partition:\n",partition)
    
    graph_a.draw_graph_communities(partition,pos=pos,subplot=plots[alphas.index(alpha)+1])
    VIC_Original[alphas.index(alpha)] += VIC(partition,partitions[SDA])
        
  
  np_alphas = np.array(alphas)
  #VIC_Diff = VIC_Configs - VIC_Original
  
  #summary[0].scatter(np_alphas,VIC_Configs,c="black",edgecolors='none',label="CONFIG")
  plots[0].scatter(np_alphas,VIC_Original,c="red",edgecolors='none',label="ORIGINAL")
  
  plt.show()
  
  robustness = np.sum(VIC_Original)
    
  return robustness, VIC_Original #VIC_Configs, VIC_Original, VIC_Diff

# Convenience function for diagnostics on sparse matrix functions
def analyze_sparse(in_am,print_whole=True):

  N = in_am.get_shape()[0]
  Deg = in_am.getnnz() # total degree
  Str = in_am.sum() # total strength 
  s_i = np.array(in_am.sum(0))[0] # strength senquence
  d_i = np.array([in_am.getcol(i).getnnz() for i in range(N)])
  
  print(N,Deg,Str)
  print(np.around(s_i,2))
  print(d_i)
  if print_whole:
    print(np.around(in_am.todense(),2))  
  else:
    print(np.around(in_am.data[0:15],2))
  
def tests():

  print("Running tests")
  # small_graph = nx.erdos_renyi_graph(25,.3)
  # small_g = Experiment_Graph("Erdos_Renyi_test",nxgraph=small_graph)
  
  # seed_set = np.zeros(25)
  # seed_set[0:4] = np.ones(4)
  # LT_params = {"seed_set":seed_set, "a_threshold":2, "type":"absolute", "debug":True}
  # small_g.LT(**LT_params)
  # LT_params = {"seed_set":seed_set, "r_threshold":.5, "type":"relative", "debug":True}
  # small_g.LT(**LT_params)
  # LT_params = {"seed_set":seed_set, "a_threshold":2, "r_threshold":.5, "type":"mixed", "debug":True}
  # small_g.LT(**LT_params)
  # # small_g.nx_IC(**params)
  
  # test_g = nx.Graph()
  # test_g.add_edges_from([(0,1),(1,2),(1,2),(0,2),(0,3),(2,3),(3,4),(4,5),(3,5),(0,6),(5,6)])
  # test_g = Experiment_Graph("test g",nxgraph=test_g)
  # seed_set = np.zeros(7)
  # seed_set[0:1] = np.ones(1)
  # LT_params = {"seed_set":seed_set, "r_threshold":.5, "type":"relative", "debug":True}
  # test_g.LT(**LT_params)

  # test_graph = nx.erdos_renyi_graph(1000,.3)
  # exp_g = Experiment_Graph("Erdos_Renyi_test",nxgraph=test_graph)
  # seed_set = np.zeros(1000)
  # seed_set[0:50] = np.ones(50)
  # LT_params = {"seed_set":seed_set, "a_threshold":2, "r_threshold":.5, "type":"mixed"}
  # print("Test Start")
  # s = time.time()
  # total = 0
  # for i in range(50):
    # total += exp_g.LT(**LT_params)[0]
  # e = time.time()
  # print("Time for 50 iterations on .IC:",round(e-s,2),"s. Avg inf:",round(total/50,1))
 
  # s = time.time()
  # total = 0
  # for i in range(50):
    # total += exp_g.nx_IC(**params)[0]
  # e = time.time()
  # print("Time for 50 iterations on nx_IC:",round(e-s,2),"s. Avg inf:",round(total/50,1)) 


if __name__ == '__main__':
  tests()
