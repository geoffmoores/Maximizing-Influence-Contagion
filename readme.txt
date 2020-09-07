The code contained in these files is the work of Geoffrey Moores in support
of my thesis "MAXIMIZING INFLUENCE OF SIMPLE AND COMPLEX CONTAGION ON 
REAL-WORLD NETWORKS". My studies were supported by the US Army, by whom I am
employed, and the National Science Foundation, who fund my research through
a Graduate Research Fellowship. I made use of many publicly available scientific
libraries, and highlight other contributions in their place in the code.
In particular, there is heavy use of the following projects, to whose authors
I offer my sincere thanks:

Graph-Tool - Tiago P. Peixoto. The graph-tool python library. 2014.
iGraph - Gabor Csardi, Tamas Nepusz, et al. The igraph software package for complex
    network research. InterJournal, complex systems, 1695(5):1{9, 2006.
NetworkX - Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring 
    network structure, dynamics, and function using NetworkX”, in Proceedings 
    of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught,
    and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
NumPy - Travis E. Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
Scipy - Travis E. Oliphant. Python for Scientific Computing, Computing in Science & 
  Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58 
Matplotlib - John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science &
  Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55 (publisher link)
  
All code is offered with the intent to help understand, reproduce, or extend the work
of my thesis, or, if it can be of other use, for whatever a reader finds it useful for. 
It is offered without any guarantees or assurances.  Where feasible, I made the code 
general, but within the scope of anticipated use cases for my thesis.  It is not a 
rigorously tested or quality controlled production outside this scope. Please feel free
to use but at your own risk.  I am happy to answer questions or engage discussion at
my personal email, geoffreymoores@hotmail.com.

A summary of the code follows, and function / class signatures and explanation
can be found in documentation.py. 

Core code
  experiment_graph.py : host of functionality and base class for all graphs explored
  simple_GA.py : a flexible genetic algorithm implementation for network optimization
  graph_greedy.py : a simple greedy algorithm (with optional submodular CELF optimization)
                for network problems
  hybrid.py : a hybrid algorithm for network optimization problems (see Ch 4 of my thesis)
  graph_gen.py : utility for generating simple artificial graphs
  general_sbm.py : more tunable utility for generation what we call "Community Graphs" in the
                thesis, but generally applicable for many SBM generation use cases
  connections_walker.py : a probabilistic model building genetic algorithm prototype for
                          network problems.
  
Auxiliary code
  animate.py : a short script to animate a matrix evolution from connections_walker.py
  heatmap.py : slightly modified heatmaps from matplotlib documentation used in thesis results
  kMeansNew.py : k-means clustering algorithm
  poisson.py : fits a poisson distribution and returns goodness of fit
  powerlaw.py : generates or fits a powerlaw distribution and returns goodness of fit

Experimental Examples
  This folder includes some of the code used to actually run experiments from my thesis,
  and may be useful for reproduction. For inspection, interest, and convenience, it also
  houses all the graph information and pickles of a majority of my experimental results.
  Results pickles are typically dictionaries and can be iterated through to explore.
