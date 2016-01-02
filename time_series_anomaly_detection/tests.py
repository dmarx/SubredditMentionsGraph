from __future__ import division
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
    
def test__graph_jaccard__commutative():
    vals1 = [a.graph_jaccard(test_mean, g) for g in graphs]
    vals2 = [a.graph_jaccard(g, test_mean) for g in graphs]
    return all(x==y for x,y in zip(vals1, vals2))

def test__graph_jaccard__correct():
    vals1 = [a.graph_jaccard(test_mean, g) for g in graphs]
    vals2 = [6/7, 5/7, 5/7, 5/8, 5/8]
    return all(x==y for x,y in zip(vals1, vals2))
    
def test__mean_graph_jaccard():
    for t in range(len(graphs)):
        v1 = a.mean_graph_jaccard(graphs, target_ix=t)['ged']
        v2 = a.graph_jaccard(test_mean, graphs[t])
        if v1 != v2:
            print t
            return False
    return True
    
def test__build_nodelist():
    nodes = a.build_nodelist(graphs)
    nodes.sort()
    vals = ['a','b','c','d','e','f','g','h','i','j','x','y','z']
    return nodes == vals