import sys
import time
import numpy as np
import random as r
import networkx as nx
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from sklearn.metrics import mutual_info_score as MI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from scipy.sparse import csr_matrix
import warnings
import scipy.stats


import experiment_graph as eg
import simple_GA as sga
import general_sbm as g_sbm

# Implementation of Multi-layer Mixed Community SBM 
# variational approximate recovery function


# IN: in_array - (2+)D array 
#     compare_axis - int, in dimensions of in_array
# OUT: out_array - same dimensions as in_array, all elements 0 except
#                  the maximum element set to one along compare_axis
def get_array_1_at_Max(in_array,compare_axis):
  
  maxima = np.amax(in_array,axis=compare_axis,keepdims=True)
  maxima = np.full(in_array.shape,maxima)
  out_array = in_array - maxima
  out_array = np.where(out_array == 0, 1, out_array*0)
  out_array = np.nan_to_num(out_array)
  
  return out_array

## CLASS ##
# Mixture_State(graph, layers, layer_width, comm_membership_info = np.array([]), 
#    comm_affiliation_info = np.array([]), edge_membership_info = np.array([]), node_membership_info = np.array([]))
#
#    graph: networkx Graph()
#    layers: int, number of layers in multi-layer sbm
#    layer_width: int, number of communities per layer
#    (comm_membership_info): (layers, layer_width) np.array, starting comm_membership_info if desired
#    (comm_affiliation_info): (layers, layer_width, layer_width) np.array, starting comm_affiliation_info if desired
#    (edge_membership_info): (self.N, self.N, layers) np.array, starting edge_membership_info if desired
#    (node_membership_info): (N, layers, layer_width) np.array, starting node membership if desired

## FIELDS ##
#  .N = # vertices
#  .M = # edges
#  .pos = nx.spring_layout(graph)
#  .layers = layers
#  .LW = layer_width
#  .adj_mat = nx.to_numpy_array(graph) #, weight=None)
#  .adj_mat *= diagonal_mask
#  .adj_mat = self.adj_mat / np.amax(self.adj_mat) # normalize weights to max of 1
#  .graph = nx.from_numpy_matrix(self.adj_mat)

#  .comm_membership: (layers, layer_width) np.array, estimated probability for community 
#                     membership, by layer
#  .comm_affiliation: (layers, layer_width, layer_width) np.array, estimated probability 
#                     for edge between nodes of communities, by layer
#  .edge_membership: (self.N, self.N, layers) np.array, estimated layer of each edge (i.e. which 
#                     layer the edge belongs to)
#  .node_membership: (N, layers, layer_width) np.array, estimated community of each node, by layer

class Mixture_State:

  def __init__(self, graph, layers, layer_width, comm_membership_info = np.array([]), 
    comm_affiliation_info = np.array([]), edge_membership_info = np.array([]), node_membership_info = np.array([])):
    
    self.N = graph.order()
    self.M = graph.size()
    self.pos = nx.spring_layout(graph)
    self.layers = layers
    self.LW = layer_width
    self.adj_mat = nx.to_numpy_array(graph) #, weight=None)
    diagonal_mask = np.ones((self.N,self.N)) - np.diag(np.ones(self.N)) # get rid of self edges
    self.adj_mat *= diagonal_mask
    self.adj_mat = self.adj_mat / np.amax(self.adj_mat) # normalize weights to max of 1
    self.graph = nx.from_numpy_matrix(self.adj_mat)
    
    self.node_membership = np.random.uniform(0,1,(self.N, layers, layer_width))
    for i in range(self.N):
      for layer in range(self.layers):
        self.node_membership[i,layer] = np.random.dirichlet(np.ones(self.LW))
    
    self.community_membership = np.full((layers, layer_width),1/layer_width)    #    Pi_(h,k)
    
    # Assuming associativity to start seems to work best
    self.community_affiliation = np.full((layers, layer_width, layer_width), .20)
    for layer in range(self.layers):
      self.community_affiliation[layer] += np.diag(np.ones(self.LW)*.5)
    #self.community_affiliation = np.random.uniform(low=0.0, high=1.0, size=(layers, layer_width, layer_width))
    
    self.edge_membership = np.full((self.N, self.N, layers),1/layers) #    e-hat_(i,j,h)

    for i in range(self.N):
      for j in range(i+1):
        if self.adj_mat[i,j] == 0:
          self.edge_membership[i,j] = np.zeros(self.layers)
          self.edge_membership[j,i] = np.zeros(self.layers)
        else:
          dirichlet_draw = np.random.dirichlet(np.ones(self.layers))*self.adj_mat[i,j]
          self.edge_membership[i,j] = np.copy(dirichlet_draw)
          self.edge_membership[j,i] = np.copy(dirichlet_draw)
          
    # Use any prior information instead, if given
    if comm_membership_info.any():
      self.community_membership = np.copy(comm_membership_info)
    if comm_affiliation_info.any():
      self.community_affiliation = np.copy(comm_affiliation_info)
    if edge_membership_info.any():
      self.edge_membership = np.copy(edge_membership_info)
    if node_membership_info.any():
      self.node_membership = np.copy(node_membership_info)
  
  # OUT: returns a new Mixture_State, deep copy, identical to self
  def copy(self):
    return Mixture_State(self.graph, self.layers, self.LW, self.community_membership, 
      self.community_affiliation, self.edge_membership, self.node_membership)
        
  # OUT: the likelihood of the graph given the current estimated parameters
  def likelihood(self):
    likelihood = 1.0
    for i in range(self.N):
      for k1 in range(self.LW):
        for layer in range(self.layers):
          z_ihk1 = self.node_membership[i,layer,k1]
          likelihood *= self.community_membership[layer,k1]**z_ihk1
          for j in range(i):
            for k2 in range(k1):
              theta = self.community_affiliation[layer,k1,k2]
              e_hat = self.edge_membership[i,j,layer]
              z_jhk2 = self.node_membership[j,layer,k2]
              likelihood *= ((theta**e_hat)*(1-theta)**(1-e_hat))**(z_ihk1*z_jhk2)
              
    return likelihood

  # OUT: the log-likelihood of the graph given the current estimated parameters
  def log_likelihood(self):

    log_likely = 0

    diagonal_mask = np.ones((self.N,self.N)) - np.diag(np.ones(self.N))
    
    for layer in range(self.layers):
      for k1 in range(self.LW):
        for k2 in range(k1):
          theta = self.community_affiliation[layer,k1,k2] # scalar
          e_hat = self.edge_membership[:,:,layer] # n x n
          z_prod= np.outer(self.node_membership[:,layer,k1], self.node_membership[:,layer,k2])*diagonal_mask

          if theta == 0:
            if len(np.where(z_prod*e_hat == 1.0)) > 0: # if there is a "certain" edge between two nodes
              return np.NINF #  which must belong to communites K1, K2, but that edge cannot exist, we return -inf. 
            log_likely += np.sum(z_prod*(1-e_hat)*np.log(1)) # otherwise we calculate with 0 weight to edge events             
          elif theta == 1:
            if len(np.where(z_prod*(1-e_hat) == 1.0)) > 0: # similarly, if there cannot be a missing edge 
              return np.NINF #  yet our model claims this edge does not exists and i/j belong exclusively to K1, K2
            log_likely += np.sum(z_prod*e_hat*np.log(1)) # otherwise we calculate with 0 weight to non edge events
          else:
            log_likely += np.sum(z_prod*(e_hat*np.log(theta) + (1-e_hat)*np.log(1-theta) ) )

    return log_likely
    
  # Not a user called method
  # calculates likelihood node in layer belongs to community, given the 
  # current estimated node membership of all other nodes for that layer
  def calc_prop(self,node,layer,community,previous_nm):
    proportion = np.log(self.community_membership[layer,community])
    
    ehats = self.edge_membership[node,:,layer]

    j_mems = np.copy(previous_nm[:,layer])  # N x K
    j_mems[node] = np.zeros(self.LW)
    thetas = self.community_affiliation[layer,community] # K
    log_thetas = np.log(np.where(thetas <= 0, 1, thetas)) # Take out all 'inexplicables'
    logmin_thetas = np.log(np.where(1-thetas <= 0, 1, 1-thetas))
    
    #proportion += np.sum((j_mems.T * ehats).T*log_thetas + (j_mems.T * (1-ehats)).T*logmin_thetas)

    one = np.outer(ehats,log_thetas)
    two = np.outer((1-ehats),logmin_thetas)
    together = one+two
    proportion += np.sum(j_mems * together)
    
    if proportion == 0:
      print("proportion was 0")
      return 0
    return np.exp(proportion)
    
  # Not a user called method
  # calculates likelihood nodes belong to all communities, by iterating
  # through a random ordering of nodes and updating estimates according to
  # all other nodes membership estimates at the previous step.
  def update_node_memberships(self):
    
    previous = np.copy(self.node_membership)

    node_shuffle = [i for i in range(self.N)]
    r.shuffle(node_shuffle)
    
    for node in node_shuffle:
      for layer in range(self.layers):
        for community in range(self.LW):
          #self.node_membership[node,layer,community] = self.calc_node_member_proportion(node,layer,community,previous)
          self.node_membership[node,layer,community] = self.calc_prop(node,layer,community,previous)
        check_zero_array(np.sum(self.node_membership[node,layer]),"1check")
        self.node_membership[node,layer] /= np.sum(self.node_membership[node,layer])
    
    # for node in range(self.N):
      # for layer in range(self.layers):
         # # Normalize to one membership / layer
        
    
    # return positive difference in new node membership scaled by total membership N * L
    delta = previous - self.node_membership
    delta = np.sum(np.where(delta < 0, delta*(-1), delta)) / (self.layers * self.N)
    
    return delta

  # Not a user called method
  # estimates prior probabilities of belonging to each community by average of current
  # node memberships
  def update_community_memberships(self):
  
    # previous = np.copy(self.community_membership)
    self.community_membership = self.node_membership.sum(axis=0) # L x LW from N x L x LW
    self.community_membership /= self.N # Normalize array s.t. the sum of any row is 1
    #                                each node contributes 1.0 total to each row, so dividing
    #                                a row, of total N, by N results in a row of sum 1.
    
  # Not a user called method
  # estimates prior probabilities of inter community edges
  # by average of current node membership estimates and the 
  # adjacency matrix
  def update_community_affiliation(self):
    
    new_comm_affiliation = np.zeros((self.layers,self.LW, self.LW))
    new_comm_denominators = np.zeros((self.layers,self.LW, self.LW))
    
    diagonal_mask = np.ones((self.N,self.N)) - np.diag(np.ones(self.N))
    
    for layer in range(self.layers):
    
      # Not sure if we can be more elegant that using these two loops
      for k1 in range(self.LW):
        for k2 in range(self.LW):
          
          NM_outer = np.outer(self.node_membership[:,layer,k2], self.node_membership[:,layer,k1])

          # Sum(over i, j) b_i,k1 x b_j,k2    <- This is our denominator to calculate Theta_layer,k1,k2
          denominator = np.sum(NM_outer*diagonal_mask)
          if denominator == 0:
            denominator = 1
            
          # Sum(over i, j) b_i,k1 x b_j,k2 * e_ij   <- This is our numerator to calculate Theta_layer,*,*
          numerator = np.sum(NM_outer*diagonal_mask*self.edge_membership[:,:,layer])
          if numerator < 0:
            print("Negative numerator in update comm affil.")
          
          new_comm_affiliation[layer,k1,k2] = numerator/denominator
          if numerator / denominator > 1:
            print(numerator, denominator, numerator/denominator)

    self.community_affiliation = np.copy(new_comm_affiliation)
    
  # not a user called method
  # updates edge membership (to which layer) by likelihood of 
  # each edge according to current node memberships and community affiliations
  def update_edge_membership(self):
    
    for layer in range(self.layers):
      for edge in self.graph.edges:
        i = edge[0]
        j = edge[1]
        product_temp = np.sum(self.community_affiliation[layer] * 
          np.outer(self.node_membership[i,layer],self.node_membership[j,layer]))
        self.edge_membership[i,j,layer] = product_temp
        self.edge_membership[j,i,layer] = product_temp
        
    # Normalize s.t. Sum over Layers : e_ij = 1 for any i,j
    self.edge_membership = norm_3D_array_3rd_axis(self.edge_membership)

    for layer in range(self.layers):
      self.edge_membership[:,:,layer] *= self.adj_mat
    
  # Utility function for presenting the current state of estimates
  def display_All(self,printing = "full", display = True):
    if printing == "full":
      print("Node membership:")
      print(np.around(self.node_membership,3))
      print("Edge membership:")
      for layer in range(self.layers):
        print(np.around(self.edge_membership[:,:,layer],3))
      print("Community Priors:")
      print(np.around(self.community_membership,3))
      print("Community affiliation:")
      print(np.around(self.community_affiliation,3))
    elif printing == "min": 
      print("Community Priors:")
      print(np.around(self.community_membership,3))
      print("Community affiliation:")
      print(np.around(self.community_affiliation,3))
      
    if display:
      fig, (comm_priors, comm_affils, node_mems, edge_mems, graph_plots) = plt.subplots(5,self.layers, gridspec_kw={'height_ratios': [.5, 1, 1, 1, 2.5]})
      fig.tight_layout()
      multiplot(self.community_membership, comm_priors, draw_array)
      multiplot(slicer(self.community_affiliation), comm_affils, draw_array)
      multiplot(slicer(self.node_membership,axis=1), node_mems, draw_array)
      multiplot(slicer(self.edge_membership,axis=2), edge_mems, draw_array)
      
      max_edges = get_array_1_at_Max(self.edge_membership,2)
      if self.layers == 1:
        draw_Graph(self.graph, max_edges[:,:,0], self.node_membership[:,0,:], subplot = graph_plots, position = self.pos)
      else:
        for index in range(self.layers):

          draw_Graph(self.graph, max_edges[:,:,index], self.node_membership[:,index,:], subplot = graph_plots[index], position = self.pos)

      draw_Graph(self.graph, self.edge_membership, [], position = self.pos)
      plt.show()
   
def check_zero_array(arg,location):
    if not isinstance(arg,np.ndarray):
      array = np.array([arg])
    else:
      array = np.copy(arg)
    if 0 in array:
      print(location, "Attempted to divide by zero.", array.shape)
      print(array)
    
def slicer(a,axis=0):
  return [r for r in np.rollaxis(a,axis)]
      
def norm_3D_array_3rd_axis(array):
  temp = np.rollaxis(array,axis=2)
  divisor = np.sum(temp,axis=0)
  divisor = np.where(divisor == 0, 1, divisor)
  temp_normed = temp / divisor
  return np.moveaxis(temp_normed, [0,1,2],[-1,0,1])
    
# Helper method for draw_Graph
def get_color(categories, colors, discrete = True):
  
  if not isinstance(categories,np.ndarray):
    array = np.array([categories])
  else:
    array = np.copy(categories)
  index = np.argwhere(array == np.amax(array))[0][0]
  if discrete:
    return colors[index]
  
  # this calculates a ratio of how far above an equal share the dominant color is, i.e. a 50% share of 2 options yields 0, but a 50% share of 4 options yields .33.  
  # full dominance (maximum is the only non zero) will always yield a 1.
  dominance = (np.amax(array) - (1 / np.size(array)) ) / (1 - (1 / np.size(array))) 
  
  color = colors[index]
  # this effect should be to darken colors which are uncertain, e.g. if a category is at or just above average it will be dark or black, but when a category is certain
  # i.e. is much greater than any other category, it should be "full" color.
  scaled_color = (color[0]*dominance,color[1]*dominance,color[2]*dominance,sc[3])
  return scaled_color
  
  # IN: graph - networkx graph
  #     edge partition - [N x N x K] np.array, K >= 1
  #     node partition - [N x K] np.array, K >= 1
  #     (subplot) - matplotlib axis for drawing the graph
  #     (discrete) - Boolean, if True then all partial memberships are completely assigned to the maximum likelihood
  #     (edge_cmap) - matplotlib colormap for edges
  #     (node_cmap) - " " "  for nodes
  #     (position) - networkx position of nodes for drawing
  # OUT: None
  # Visualizes a single layer of community assignments
def draw_Graph(graph, edge_partition, node_partition, subplot = None, discrete=True, edge_cmap = "tab10", node_cmap = "tab10", position = None):

  if subplot == None:
    fig, subplot = plt.subplots(1)
    
  plt.sca(subplot)
  
  drawn_graph = nx.Graph()
  
  # Add colored edges to drawn_graph
  edge_cmap = plt.get_cmap(edge_cmap)
  edge_categories = np.size(edge_partition[0][0])+1
  edge_colors_options = edge_cmap(range(edge_categories))

  for e in graph.edges():
    # we only want edges that exist in this partition. 
    if np.sum(edge_partition[e[0],e[1]]) > 0: 
      eij_color = get_color(edge_partition[e[0],e[1]],edge_colors_options, discrete = discrete)
      drawn_graph.add_edge(e[0],e[1],color=eij_color)

  edge_colors_list = []
  for e in drawn_graph.edges(data='color'):
    edge_colors_list.append(e[2])
  
  # Add colored nodes to drawn_graph
  if len(node_partition) > 0:
    node_cmap = plt.get_cmap(node_cmap)
    node_categories = np.size(node_partition[0])+1
    node_color_options = node_cmap(range(node_categories))
    
    node_labels = {node:node for node in drawn_graph.nodes}
    
    node_colors_list = [get_color(node_partition[node], node_color_options, discrete = discrete) for node in drawn_graph.nodes]
  else:
    node_labels = {node:node for node in drawn_graph.nodes}
    node_colors_list = ["black" for node in drawn_graph.nodes]
   
  if position == None:
    position = nx.spring_layout(drawn_graph)
  nx.draw(drawn_graph,position,edge_color=edge_colors_list,width=.5,node_size=45, node_color=node_colors_list,labels=node_labels, font_color="white", font_size = 8)
  
# Visualizes an array in grey tone
def draw_array(array, subplot=None, cmap="Greys", show=False):

	#Plot adjacency matrix in toned-down black and white
  if subplot == None:
    fig, subplot = plt.subplots(1)
  
  if array.ndim == 1:
    subplot.imshow(array.reshape(1,array.shape[0]), cmap, interpolation="none", vmin = 0, vmax = 1)
  else:
    subplot.imshow(array, cmap, interpolation="none", vmin = 0, vmax = 1)
  subplot.axis('off')
  if show:
    plt.show()

# Helper function for display_all
def multiplot(items,subplots,draw_function):
  
  num_items = len(items)
  if num_items == 1:
    draw_function(items[0],subplot=subplots)
  else:
    for index in range(num_items):
      draw_function(items[index],subplot=subplots[index])
      
# Main user called function
# IN: graph - networkx graph 
#     layers - int, number of layers in multi layer sbm
#     layer_width - int, number of communities per layer
#     (runs) - int, number of times to run recovery
#     (display) - Boolean, if True outputs a set of visualizations, one per layer and one summary
#     (printing) - String, defaults to "min", controls level of detail printed to console
#                  "full" - all matrices of Mixture_State printed
#                  else - nothing printed
#     (merge_graph) - a labeled multi layer sbm which is used as ground truth to evaluate NMI
#                     of the Mixture_State, if given
# OUT: Mixture_State - best mixture_state found across runs
#      best_likelihood - likelihood of most likely state found
def model_recover(graph,layers,layer_width,runs=1,display=False,printing="min",merge_graph=None):

  best_like = np.NINF
  
  for x in range(runs):
    
    mixer = Mixture_State(graph,layers,layer_width)

    c2_threshold = 0.0001 #  convergence threshold for node memberships, potentially expensive to set low
    max_its_2 = 25
    max_its_1 = 40
    iterations = 0

    outer_likelihood = mixer.log_likelihood()
    outer_delta_like = 1
    relative_delta = 1
    
    ts = time.time()
    while (iterations < max_its_1 and outer_delta_like > 0) or iterations < 10: 
      iterations += 1

      node_delta = 1.0
      its_2 = 0
      
      last_like = np.NINF
      mix_like = 0

      while its_2 < max_its_2 and node_delta > c2_threshold and (its_2 < 5):
        its_2 += 1
        
        node_delta = mixer.update_node_memberships()
        
        last_like = mix_like

      mixer.update_edge_membership()
      if iterations > 0:
        mixer.update_community_memberships()
        mixer.update_community_affiliation()
      
      temp = outer_likelihood
      outer_likelihood = mixer.log_likelihood() 
      outer_delta_like = outer_likelihood - temp

    te = time.time()
    print("Run:",x," iterations:",iterations," likelihood:",outer_likelihood,"in",te-ts,"seconds.")
    
    if outer_likelihood > best_like:
      best_mixer = mixer.copy()
      best_like = outer_likelihood
    
    if merge_graph != None:
      print(recovery_NMI(mixer,merge_graph))
      
  best_mixer.display_All(printing=printing, display=display)
  
  return best_mixer, best_like
  
# Same function as gen_sbm in graph_gen.py
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
  
# Calculates the Normalized Mutual information between the mixer's estimated memberships
# and the true memberships stored within merge_graph, which should be built using 
# general_sbm
def recovery_NMI(mixer,merge_graph):

  warnings.simplefilter("ignore")
  
  def comm_ind(node, layer):
    matched = [x for x in merge_graph.nodes[node]['communities'] if layer in x][0]
    return int(matched[len(layer)+1:])
    
  def greedy_max_map(length, width, scores):
    
    x = min(length, width)
    working = np.copy(scores)
    sum = 0
    max_map = []
    
    for i in range(x):
      max = np.amax(working)
      row,col = np.argwhere(working == max)[0]
      working[row] *= 0
      working[:,col] *= 0
      sum += max
      max_map.append((row,col))
      
    return sum, max_map
    
  L = mixer.layers
  LW = mixer.LW
  
  true_layers = merge_graph.graph['communities']
  LG = len(true_layers)
  true_comms = [] # LG x N
  for true_layer in true_layers:
    true_comms.append([comm_ind(n,true_layer) for n in merge_graph.nodes()])
    
  est_comms = []
  for layer in range(L):
    nm_l = mixer.node_membership[:,layer]
    est_comms.append(list(np.argmax(nm_l,axis=1)))
  
  layer_map = []
  nmi_node_layers = np.zeros((LG,L))
  
  for i in range(LG):
    for j in range(L):
      nmi_node_layers[i,j] = I_norm(true_comms[i],est_comms[j])
  
  node_nmi = greedy_max_map(LG,L,nmi_node_layers)[0]/min(L,LG)
  
  est_edge_mem = []
  true_edge_mem = []
  for e in merge_graph.edges():
    i,j = e
    e_est = mixer.edge_membership[i,j]
    e_est_layer = np.argmax(e_est)
    
    e_true_layer = r.sample(merge_graph[i][j]['communities'],1)[0]
    e_true_layer = true_layers.index(e_true_layer)
    
    est_edge_mem.append(e_est_layer)
    true_edge_mem.append(e_true_layer)
    
  edge_nmi = I_norm(est_edge_mem,true_edge_mem)
  
  return node_nmi, edge_nmi
    
def I_norm(A,B):
  return 2*MI(A,B)/(H(A)+H(B))

  # Entropy
  # In: Partition A, np.array of INT
  # Out: H(A) = - SUM_a: P(A)*log[P(A)]
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
    
def tester():
  
  # G = nx.Graph()
  # g1 = nx.complete_graph(100)
  # g2 = nx.complete_graph(100)
  # A = nx.disjoint_union(g1,g2)
  # second_layer_clique_1 = r.sample(set(A.nodes),100)
  # second_layer_clique_2 = set(A.nodes).difference(second_layer_clique_1)
  # #second_layer_clique_1 = [0,1,2,3,4,5,6,7,29,28,27,26,25,24,23]
  # #second_layer_clique_2 = [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
  # for i in second_layer_clique_1:
    # for j in second_layer_clique_1:
      # if i != j:
        # A.add_edge(i,j)
  # for i in second_layer_clique_2:
    # for j in second_layer_clique_2:
      # if i != j:
        # A.add_edge(i,j)
  
  # G = A.copy()
  
  size = 500
  num_comms = 5
  comm_size = int(size / num_comms)
  comms = [int(i/(size/num_comms)) for i in range(size)]
  results = []
  
  for b in range(0,3):
    for a in range(1,11):
    
      g1 = gen_sbm(size,num_comms,a/10,b/10,probabilities=True)
      g2 = gen_sbm(size,num_comms,a/10,b/10,probabilities=True)
      #g3 = gen_sbm(size,num_comms,1,0,probabilities=True)
      
      g_sbm.label_graph(g1.nxgraph,"A",comms)
      g_sbm.label_graph(g2.nxgraph,"B",comms)
      # g_sbm.draw_community_graph(g1.nxgraph,show=True)
      # g_sbm.draw_community_graph(g2.nxgraph,show=True)

      new_g = g_sbm.merge_graphs([g1.nxgraph,g2.nxgraph])
      new_eg = eg.Experiment_Graph("merge_i",nxgraph = new_g)
        
      mixer, likely = model_recover(new_eg.nxgraph,layers=2,layer_width=5,runs=10,display=False,merge_graph=new_g)
      
      results.append([a/10,b/10,recovery_NMI(mixer,new_g),likely])
      print(results[-1])
  
  print(results)
  
if __name__ == "__main__":  
  tester()
