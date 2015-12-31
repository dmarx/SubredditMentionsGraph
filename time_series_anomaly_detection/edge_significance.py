from __future__ import division
import networkx as nx
import numpy as np
import scipy as sp

def node_strength(g, n):
    """Returns the node strength for node n in graph g"""
    s = 0
    for target in g.neighbors(n):
        s+= g[n][target]['weight']
    return s

def normalize_graph(g):
    """Returns a new graph whose edges are normalized such that each node (with positive strength) has strength = 1"""
    #strength = [node_strength(g,n) for n in g.nodes()]
    strength = dict((n, node_strength(g,n)) for n in g.nodes())
    #if g.is_directed():
    #    gn = nx.DiGraph()
    #else:
    #    gn = nx.Graph()
    for u,v in g.edges():
        w = g[u][v]['weight']
    #    gn.add_edge(u,v, {'weight':w/strength[u]})
        g[u][v]['normalized_weight'] = w/strength[u]
    #return gn
    return g

def edge_significance(p_ij, k):
    def integrand(x):
        return np.power(1-x, k-2)
    return 1 - (k-1) * sp.integrate.quad(integrand, 0, p_ij)[0]

def filter_graph(g, alpha, return_filtered_copy=True):
    g = normalize_graph(g)
    if return_filtered_copy:
        if g.is_directed():
            g2 = nx.DiGraph()
        else:
            g2 = nx.Graph()
    for u,v in g.edges():
        w = g[u][v]['normalized_weight']
        d = g.degree(u)
        p = edge_significance(w, d)
        g[u][v]['significance'] = p
        g[u][v]['is_significant'] = p<alpha
        if return_filtered_copy and p<alpha:
            g2.add_edge(u,v, {'weight':w, 'significance':p})
    retval = {'graph':g}
    if return_filtered_copy:
        retval['filtered_graph'] = g2
    return retval