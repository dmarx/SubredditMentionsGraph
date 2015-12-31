import cPickle
import time
from anomaly import *

with open('timeslice_graphs_filtered.dat', 'r') as f:
    unp = cPickle.Unpickler(f)
    filtered_graphs = unp.load()

def ged_all_iteration(graphs, radius = 1, target_ix=-1, score_as_ratio=True, target_is_denominator=False, verbose=False):
    nodes = graphs[target_ix].nodes()
    score = {}
    n = nodes[0]
    if verbose: print n
    #subgraphs = [nx.ego_graph(g, n, radius) for g in graphs]
    subgraphs = []
    for g in graphs:
        if g.has_node(n):
            #sub_g = nx.ego_graph(g, n, radius)
            start = time.time()
            sub_g = directed_ego_graph(g, n, radius)
            end = time.time()
            print "ego graph:", end-start
        else:
            sub_g = nx.Graph()
        subgraphs.append(sub_g) # Overlapping subproblems for separate ged_all calls!!!
    start = time.time()
    if score_as_ratio:
        score[n] = mean_graph_ged_ratio(graphs, target_ix,  
                        target_is_denominator=target_is_denominator)
    else:
        score[n] = mean_graph_ged(graphs, target_ix)
    end = time.time()
    print "scoring:", end - start # Nearly all time is spent scoring. This should be fast. 
    return score
    
#test = ged_all(filtered_graphs[:5], verbose=True)
#test = ged_all_iteration(filtered_graphs[:5], verbose=True)

import cProfile

cProfile.run('test = ged_all_iteration(filtered_graphs[:5], verbose=True)') #.672, .513, .693
#cProfile.run('test = ged_all_iteration(filtered_graphs[:5], verbose=True, score_as_ratio=False)')
##########################

start = time.time()
construct_mean_graph(filtered_graphs[:5])
end = time.time()
print "csr_matrix:", end - start

start = time.time()
construct_mean_graph(filtered_graphs[:5], format='coo')
end = time.time()
print "coo_matrix:", end - start



# csr_matrix is slow, but it's still about 30% faster than the counter method.

##############################

g1 = directed_ego_graph(filtered_graphs[0], 'askreddit', radius=1)
g2 = directed_ego_graph(filtered_graphs[1], 'askreddit', radius=1)

nodes = set(adj1.nodes())
nodes.update(adj2.nodes())

adj1 = nx.to_scipy_sparse_matrix(g1, nodelist=nodes)
adj2 = nx.to_scipy_sparse_matrix(g1, nodelist=nodes)

%timeit naive_graph_edit_distance_adj(adj1, adj2)