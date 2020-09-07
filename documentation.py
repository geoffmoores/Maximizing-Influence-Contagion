experiment_graph.py

## CLASS ##
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

## METHODS ##

  # IN: am - csr_matrix
  # OUT: None
  # sets the library specific graph objects according to am
  def set_graphs_adj(self,am):

  # IN: am - csr_matrix 
  # OUT: None
  # .gtgraph populated from adj_mat
  def gt_from_adj(self,adj_mat):

  # IN: am - csr_matrix 
  # OUT: None
  # .nxgraph populated from am
  def nx_from_adj(self,am):

  # IN: am - csr_matrix 
  # OUT: None
  # .igraph populated from am
  def ig_from_adj(self,am):

  # IN: g - graph-tool Graph object
  # OUT: csr_matrix, adjacency matrix data from g
  # Warning: this will memory hog as is, but currently not used
  # TODO: adjust for efficient usage of csr_matrix
  def adj_from_gt(self,g):

  # IN:  (alpha) - Real number [0,1], proportion of edges to rewire
  #      (ret_Exp_Graph) - Boolean, whether or not to return a new object or simply mutate the calling object
  # OUT: Experiment_Graph object - if ret_Exp_Graph == True, returns a new object
  
  # Generates a configuration graph from self, a proportion of edges equal to alpha are randomly 
  # rewired (with replacement)
  # NOTE: Only works for unweighted graphs.  Should not be used by default.
  def config_graph(self,alpha=1.0,ret_Exp_Graph=False):

  # IN: None
  # Out: None
  # adj_mat is normalized, then all edge weights are shifted such that
  # the minimum edge weight is 0. 
  def norm_pos_adj_mat(self):

  # IN: None
  # Out: None
  # all edge weights are shifted such that the minimum edge weight is 0
  # and then all weights are divided by the maximum weight
  def scale_pos_adj_mat(self):

  # IN: i - integer in [0,self.N-1] of node of interest
  # OUT: degree(v_i) - int
  # Caches degree sequence in temp["deg_seq"] 
  def get_degree(self,i):

  # IN: (relative) - Boolean, if true the returned array is divided by the total number of edges
  # OUT: degree sequence - 1D np.array (ints), the degree sequence of the graph
  # Caches result in temp["deg_seq"] 
  def get_deg_seq(self,relative=False):

  # IN: i - integer in [0,self.N-1] of node of interest
  # OUT: strength(v_i) - float, the sum of edge weights
  # Caches strength sequence in temp["str_seq"] 
  def get_strength(self,i):

  # IN: (relative) - Boolean, if true the returned array is divided by the sum of edge weights
  # OUT: strength sequence - 1D np.array, the strength sequence of the graph
  # Caches result in temp["str_seq"] 
  def get_str_seq(self,relative=False):

  # IN: (debug) - Boolean, deprecated (no effect)
  #     (recalc) - Boolean, if False then the cached version of distances will be overwritten
  # OUT: 2D np.array, distances in hops between all node pairs in the graph (np.Inf if no path exists)
  # Stores result in self.distances
  def get_distances(self,debug=False,recalc=False):
  
  # IN:  (alpha) - Real number [0,1], proportion of edges to rewire
  #      (ret_Exp_Graph) - Boolean, whether or not to return a new object or simply mutate the calling object
  # OUT: Experiment_Graph object - if ret_Exp_Graph == True, returns a new object
  
  # Generates a configuration graph from self, a proportion of edges equal to alpha are randomly 
  # rewired (with replacement). Preserves (in expectation) the degree and strength of all nodes.
  # This is an extension of the work of Palowitch et al "Significance-based community detection in weighted networks"
  def wtd_config_graph(self,alpha=1.0,ret_Exp_Graph=True):
  
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
  
  # Alternate IC implementation which uses sets and the networkx graph instead
  # of the adjacency matrix to simulate process spread. Faster if the expected
  # number of nodes infected each timestep is very small and the graph is large.
  # Shares the same parameters as IC(), but can handle directed graphs. 
  # Unfortunately, self.nxgraph is not a directed graph by default, so to use you 
  # will have to make your own adjustments (or just extract this function). 
  def nx_IC(self,**kwargs):
  
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
  
  # IN: **kwargs
  #   Mandatory:
  #     "seed_set" - 1D np.array, length N, where 1 -> node @ index is a seed
  #     "adv_set" - 1D np.array, length N, where 1 -> node @ index is an adversarial seed
  #     "type" - "relative" : threshold is considered a relative fraction of infected neighbors
  #              "absolute" : threshold is considered an absolute weight or number of infected neighbors
  #              else : threshold is the minimum of the relative and absolute thresholds provided (per node)
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
  # NOTE: infection factor is hard coded for all experiments to an estimate from Ugander et al's 
  #         "Structural diversity in social contagion" paper, but can be set as desired to tune behavior.
  def UC(self,**kwargs):
  
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
  
  # IN: (am) - np.array, if provided uses the given array instead of self.adj_mat
  # OUT: None
  # Prints some simple measures of a 2D array interpreted as an adjacency matrix.
  def analyze_am(self, am=np.array([])):
  
  # IN: structure_label - String, key for sructure to be stored
  #     structure - a structure of the graph (e.g. a partition from community detection)
  # OUT: None
  # self.structures updated
  def add_structure(self,structure_label,structure):
    self.structures[structure_label] = structure  
  
  # IN: None
  # OUT: Returns most likely SBM recovered between the Classic or Degree-Corrected models
  def best_SBM(self):
  
  # Convenience function for plotting robustness as it is stored by structure_test
  def plot_robustness(self,subplot=None):
  
  # IN: (path)
  # OUT: None
  # pickles the Experiment_Graph object at path (or self.path) 
  def pickle(self,path=""):
  
  # IN: structure_detect_alg - String {"sbm","dc_sbm","infomap"} determines which community detection alg to run
  #     (runs) - integer number of times to execute the detection algorithm
  #     (add) - Boolean, whether to save the partition in self.structures
  #     (reset) - Boolean, whether to ignore any previously saved structures
  # OUT: structure as returned according to the structure detection algorithm selected
  def get_structure(self,structure_detect_alg,runs=5,add=False,reset=False):
  
  # IN: structure - object returned by infomap or SBM recovery
  #     SDA - String {"sbm","dc_sbm","infomap"}
  # OUT: 1D np.array, partition of nodes according to structure given
  def partition(self,structure,SDA):  
 
  # IN: structure - object returned by infomap or SBM recovery
  #     SDA - String {"sbm","dc_sbm","infomap"}
  # OUT: real number, modularity of partition according to structure given
  def modularity(self,structure,SDA): 
  
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
  
  # Convenience method to set a single graph layout for consistent drawing.
  def set_pos(self):
  
  # IN: partition - 1D np.array of node community affiliations
  #     (pos) - networkx graph layout for drawing
  #     (subplot) - matplotlib axis to draw in 
  # OUT: None
  # Draws the graph with nodes colored by the community affiliations in partition
  def draw_graph_communities(self,partition,pos=None,subplot=None):
 
## NON-CLASS METHODS ##

# In: Two partitions, A, B, which are np.arrays of INT of equal length
# Out: V(A,B) = H(A) + H(B) - 2I(A,B), a number in 0-1
# Normalized Variation of Information
def VIC(A,B):

# In: Two partitions, A, B, which are np.arrays of INT of equal length
# Out: V(A,B) = H(A) + H(B) - 2I(A,B), a number in 0-1
# Normalized Mutual Information
def I_norm(A,B):

# In: Partition A, np.array of INT
# Out: H(A) = - SUM_a: P(A)*log[P(A)]
# Entropy of list/array A
def H(A):

# Returns a sparse matrix with only the negative elements in cm
def get_csr_neg(cm):

# Returns a sparse matrix with only the positive elements in cm 
def get_csr_pos(cm):

# IN: filename - String, filename of a .edges file to read from
# OUT: edges - 2-column np.array, each row is a node pair denoting edges in a graph
#      weights - 1D np.array(float or integer) denoting edge weights
# NOTE: some regular expression work was done to make this work with a majority of 
#      .edges files from NetworkRepository.com. It should work on most lists of edges,
#      in particular any of the following forms, where the square brackets denote 
#      the format of each line, and the spaces can be any whitespace characters:
#      [int int], [int int float], [int int float ******]
def read_edges(graph_filename):

# IN: graphlist_file - list(String), filenames of .edges files to read from
#     (reload) - Boolean, whether to read from .edges file or attempt to load a pickle first
# OUT: list(Experiment_Graph) - all experiment graphs successfully loaded from graphlist_file
# NOTE: 1) filepath is currently hardcoded
def read_graphs_from_list(graphlist_file,reload=False):

# IN: graph_name - String, filename of a .edges file to read from
#     graph_type - String, category and subfolder on hardcoded path below to find .edges file
#     (reload) - Boolean, whether to read from .edges file or attempt to load a pickle first
# OUT: experiment graph - Experiment_Graph if successfully loaded from path+graph_type+graph_name
# NOTE: 1) filepath is currently hardcoded
def load_graph(graph_name,graph_type,reload=False):

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
  
# This is a visualization method which displays the communities detected as the graph
# is perturbed according to alphas, using a single structure / community detection algorithm SDA. 
def inspect_robust(graph,SDA,alphas=[i*.05 for i in range(6)],runs=5):

# Convenience function for diagnostics on sparse matrix functions
def analyze_sparse(in_am,print_whole=True):

general_sbm.py
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

## METHODS ##

  # IN: graph - networkx graph object
  # OUT: None
  # Applies the SBM object's edge and group distributions to the graph,
  # storing community assignments in the networkx graph's node and edge dictionaries
  def apply(self,graph):
  
  # IN: graph - networkx graph object
  # OUT: dictionary[int] -> int, community assignments for all nodes
  def apply_communities(self,graph):
  
  # IN: graph - networkx graph
  #     temp_comms - dictionary partition of nodes to communities
  # OUT: None
  # graph mutated with added edges and stored community assignments
  def apply_edges(self,graph,temp_comms):  

## NON-CLASS METHODS ##

# IN: graph - networkx graph object
#     name - string for community name
#     communities - list(int) length |V| of graph
# OUT: None
# Labels a graph, without use of an sbm object, that already has edges
# according to the community assignments in communities
def label_graph(graph, name, communities):  

# IN: graphs - list(networkx graphs), each must have an sbm applied
#     (scramble) - Boolean, toggle whether to decouple node indices between graphs
# OUT: networkx graph, with the union of all sbm information from all in graphs 
def merge_graphs(graphs,scramble=True):

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
 
multi_layer_sbm_recover.py

# Main user called function:
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

## NON-CLASS METHODS ##
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
 

# Calculates the Normalized Mutual information between the mixer's estimated memberships
# and the true memberships stored within merge_graph, which should be built using 
# general_sbm
def recovery_NMI(mixer,merge_graph): 
 
graph_gen.py

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

simple_GA.py
# Provides general functionality for modular genetic algorithms.
# Given parameters, reproduction, mutation, and parent selection functions,
# GA_generator returns a function which when called executes the defined
# genetic algorithm and returns solutions and fitness over time. 
# NOTE: When run as main, simple_GA.py conducts a series of tests for individual
#       operators and genetic algorithms. This behavior can be changed at the end
#       of the script.


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
#     genetic_algorithm() returns:
#       best_individual - best solution found by the GA
#       total_process_evals - integer, number of times fitness function was called 
#       fitness_over_time - list(float), best / elite fitness over GA generations 
#       avg_fitness_over_time - list(float), average fitness over GA generations 
#       pool - list([individual,float]), all individuals and their fitness from final generation
def GA_generator(fitness_evaluator, individual_generator,
  ga_params = {"pr_mutate":.01,"iterations":100,"pool_size":100,"tournament_size":3},
  mutator=default_mutator, reproducer=default_reproducer, parent_selector=default_parent_selector,
  constraint_checker=default_constraint_checker,fitness_processes_per_eval=1,num_elites=2,multiprocess=False,
  debug=False,status=True,early_stop=True)
  
## NETWORK SPECIFIC GENETIC OPERATORS ##
# IN: pool - list([individual,fitness]), genetic algorithms pool
#     (size) - integer > 0, number of parents to select
def fit_prop_select(pool,size=2):

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

# IN: graph - Experiment_Graph object
#     trials - integer # of trials to evaluate fitness
#     process - a function of graph from (individual) -> float
#     process_params - a dictionary of parameters for process
#     k - size of individuals in # of nodes
# OUT: nlist_fitness(individual,scaling_factor,fit_dict)
#         IN: individual - length k solution
#             scaling_factor - float in [0,1], closeness to end of GA, allows for temporal selection pressure
#                              on oversize solutions (eventually nlist should develop solutions of length k)
#             fit_dict - dictionary[individual] -> float, cache of known fitnesses to spare calculation
#         OUT: fitness of individual, float
def nlist_fit_wrapper(graph,trials,process,process_params,k):

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

# compact generator for standard GA's on networks
# minimal example call: 
#      simple_GA.K_graph_GA(my_graph,my_graph.LT,{"seed_set":None, "a_threshold":2, "type":"absolute"})
def K_graph_GA(graph,process,process_params,trials=1,k=None,mutator="random",crossover="single_point",pc=.3,deg_wtd=False, early_stop=True,
    GA_params={"pr_mutate":.8,"iterations":100,"pool_size":100,"tournament_size":2},status=False):
    
graph_greedy.py
# IN: graph - Experiment_Graph
#     fun - function(networkx graph, list(node indices)) -> float
#     k - list(int), sizes of solutions desired
#     (submodular) - Boolean, if true will use the CELF optimization
#     (progress) - Boolean, if true will print progress statements during execution
# OUT: results - list(lists(int)), set of solutions, one for each size in k
#      evals - integer, number of times fun invoked
def greedy(graph,fun,k,submodular=False,progress=False):

connections_walker.py
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
  
hybrid.py
# hybrid.py attempts to maximize some function of a set of nodes on a graph
# by first partitioning the graph. Then two independent methods determine where
# to allocate the K nodes among the partitions and where to allocate the nodes
# assigned to each partition, respectively. (See sec 4.2 of my thesis if this is unclear). 

# IN: graph - Experiment_Graph object
#     process - function(list(int),graph) -> float, a function to maximize
#     ks - list(int), sizes of solutions desired
#     outter - "ga" : uses a genetic algorithm to maximize the assignment of nodes across partition communities
#              else : uses a greedy algorithm  " " "
#     inner - "ga"  : uses a genetic algorithm to maximize the assignment of nodes within a single community
#             else  : uses a greedy algorith " " " 
#     partitioner - "sbm" : partitions the graph according to SBM or DC_SBM, whichever has a better fit
#                   else  : partitions the graph using infomap
#     **kwargs - "submod" : uses a submodular greedy algorithm for the inner maximizations
# OUT: total_solutions - list(list(int)), solutions according to sizes in ks
#      out_evals - integer, number of times process was called on the outter solutions
#      inner_evals - integer, number of times process was called on the inner solutions
def run_hybrid(graph,process,ks,outter,inner,partitioner,**kwargs):

heatmap.py

def category_map(data, row_labels, col_labels, categories, cmap="tab10", ax=None, cmax=None,
            cbar_kw={}, cbarlabel="", title = "Default",  x_title=" ",y_title=" ", saveFile = None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M), values in [0,len(categories)-1].
    categories
        a list of category labels the indices in data correspond to
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", title = "Default", x_title=" ",y_title=" ",saveFile = None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
powerlaw.py

# IN: data - list(int)
#     (K_min) - minimum value K to begin the powerlaw fit at
#     (get_goodness_of_fit) - Boolean, if true calculates and returns Goodness of Fit 
#                             with powerlaw parameters
#     (get_var) - Boolean, if true returns the variance of the fitted powerlaw distribution
def fit(data,K_min=1,get_goodness_of_fit=True,get_var=True):

# IN: gamma - float in [2,3] preferably, degree exponent of distribution
#     kmin - minimum degree generated in sequence
#     kmax - maximum degree generated
#     N - number of samples
# OUT: 1D np.array length N, powerlaw of degrees according to parameters given
def powerlaw_degrees(gamma,kmin,kmax,N):

poisson.py

# IN: data - 1D np.array or list of data 
# OUT: mu - float, expected average value of fit Poisson distribution
#      goodness of fit - float in [0,1], p-value of poisson distribution fit with observed data
#                        according to chi-square test.
def fit(data):