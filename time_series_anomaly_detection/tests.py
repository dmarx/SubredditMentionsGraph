import anomaly as a

import networkx as nx

# Set up some test cases
g1 = nx.DiGraph()
g2 = nx.DiGraph()
g3 = nx.DiGraph()
g4 = nx.DiGraph()
g5 = nx.DiGraph()

g1.add_edges_from(
    [
     #('z','a',{'weight':1}),
     ('z','b',{'weight':1}),
     ('z','c',{'weight':1}),
     ('z','d',{'weight':1}),
     ('z','e',{'weight':1}),
     ('z','f',{'weight':1}),
     ('z','x',{'weight':1}),
     ('y','z',{'weight':1})
    ])
    
g2.add_edges_from(
    [
     #('z','a',{'weight':1}),
     #('z','b',{'weight':1}),
     ('z','c',{'weight':1}),
     ('z','d',{'weight':1}),
     ('z','e',{'weight':1}),
     ('z','g',{'weight':1}),
     ('z','x',{'weight':1}),
     ('y','z',{'weight':1})
    ])

g3.add_edges_from(
    [
     #('z','a',{'weight':1}),
     ('z','b',{'weight':1}),
     #('z','c',{'weight':1}),
     ('z','d',{'weight':1}),
     ('z','e',{'weight':1}),
     ('z','h',{'weight':1}),
     ('z','x',{'weight':1}),
     ('y','z',{'weight':1})
    ])
    
g4.add_edges_from(
    [
     ('z','a',{'weight':1}),
     ('z','b',{'weight':1}),
     ('z','c',{'weight':1}),
     #('z','d',{'weight':1}),
     ('z','e',{'weight':1}),
     ('z','i',{'weight':1}),
     ('z','x',{'weight':1}),
     ('y','z',{'weight':1})
    ])
      
g5.add_edges_from(
    [
     ('z','a',{'weight':1}),
     ('z','b',{'weight':1}),
     ('z','c',{'weight':1}),
     ('z','d',{'weight':1}),
     #('z','e',{'weight':1}),
     ('z','j',{'weight':1}),
     ('z','x',{'weight':1}),
     ('y','z',{'weight':1})
    ])
    
graphs = [g1, g2, g3, g4, g5]
    
test_mean = nx.DiGraph()
test_mean.add_edges_from([
    ('z','b',{'weight':1}),
    ('z','c',{'weight':1}),
    ('z','d',{'weight':1}),
    ('z','e',{'weight':1}),
    ('z','x',{'weight':1}),
    ('y','z',{'weight':1})
    ])
    
##########################

def test__construct_mean_graph__adj():
    mean_adj = a.construct_mean_graph(graphs)
    
    nodes = a.build_nodelist(graphs)
    test_adj = nx.to_scipy_sparse_matrix(test_mean, nodelist=nodes, format='csr')
    comp = mean_adj.multiply(test_adj)
    
    return comp.sum() == mean_adj.sum() == test_adj.sum()

def test__construct_mean_graph__graph():
    mean_g = a.construct_mean_graph(graphs, as_adjacency=False)
    return set(list(mean_g.edges())) == set(list(test_mean.edges()))