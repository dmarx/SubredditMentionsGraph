#cd E:\Projects\subreddit_mentions_graph
from __future__ import division
import sqlite3
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from edge_significance import filter_graph
from collections import deque, Counter
from sklearn.preprocessing import binarize
from itertools import dropwhile

def get_time_graph_from_df(df, t0, t1, agg='distinct'):
    """agg must be one of distinct, count, or sum"""
    subset = df[np.logical_and(t0 <= df.date, df.date < t1)]
    if agg == 'distinct':
        grouped = subset.groupby(['source','target'], as_index=False).agg({"weight":pd.Series.nunique})
    elif agg == 'count':
        grouped = subset.groupby(['source','target'], as_index=False).count()
    elif agg == 'sum':
        grouped = subset.groupby(['source','target'], as_index=False).sum()
    else:
        raise Exception("Unsupported aggregation: {}".format(agg))
    #grouped.reset_index(inplace=True)
    g = nx.DiGraph()
    g.add_weighted_edges_from(grouped.values)
    return g
    #return grouped

# This would be better if it was generalized to operate on a pd.DataFrame
# with source/target/date columns. This database IO is killing me.
def get_time_graph(conn, t0, t1):
    df = pd.read_sql("""
        SELECT  LOWER(source_subr) source,
                LOWER(target_subr) target,
                COUNT(DISTINCT author) weight
        FROM    mentions
        WHERE   LOWER(source_subr) <> LOWER(target_subr)
        AND     created_utc BETWEEN {t0} AND {t1}
        GROUP BY LOWER(source_subr), LOWER(target_subr)
        --HAVING COUNT(DISTINCT author) > 2
        """.format(t0=t0,t1=t1), 
        conn)
    #return df
    g = nx.DiGraph()
    g.add_weighted_edges_from(df.values)
    return g
        
def get_nodes_list(conn):
    df = pd.read_sql("""
        SELECT  LOWER(source_subr) source,
                LOWER(target_subr) target,
                COUNT(DISTINCT author) weight
        FROM    mentions
        WHERE   LOWER(source_subr) <> LOWER(target_subr)
        GROUP BY LOWER(source_subr), LOWER(target_subr)
        --HAVING COUNT(DISTINCT author) > 2
        """, conn)
    return df.source.append(df.target).unique()

def get_time_graphs_range(conn, 
                          t_start, 
                          t_end, 
                          interval_length=30*(60*60*24),
                          delta=30*(60*60*24),
                          verbose=False
                          ):
    t0 = t_start
    t1 = t0 + interval_length
    graphs = []
    while t1 < t_end:
        if verbose:
            print t0, t1
        g = get_time_graph(conn, t0, t1)
        graphs.append((t0, t1, g))
        t0 = t0 + delta
        t1 = t0 + interval_length
    return graphs
    
def get_time_graphs_range_from_df(df, 
                          t_start, 
                          t_end, 
                          interval_length=30*(60*60*24),
                          delta=30*(60*60*24),
                          verbose=False
                          ):
    t0 = t_start
    t1 = t0 + interval_length
    graphs = []
    while t1 < t_end:
        if verbose:
            print t0, t1
        #g = get_time_graph(conn, t0, t1)
        g = get_time_graph_from_df(df, t0, t1)
        graphs.append((t0, t1, g))
        t0 = t0 + delta
        t1 = t0 + interval_length
    return graphs
    
def single_source_shortest_path_length(G,source,cutoff=None, direction="forwards"):
    """Compute the shortest path lengths from source to all reachable nodes.
    
    Adapted from nx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length 
    to include a direction argument.
    
    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.
    
    direction: Whether to calculate forwards paths or reverse paths.
        
    Returns
    -------
    lengths : dictionary
        Dictionary of shortest path lengths keyed by target.

    Examples
    --------
    >>> G=nx.path_graph(5)
    >>> length=nx.single_source_shortest_path_length(G,0)
    >>> length[4]
    4
    >>> print(length)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    See Also
    --------
    shortest_path_length
    """
    seen={}                  # level (number of hops) when seen in BFS
    level=0                  # the current level
    nextlevel={source:1}  # dict of nodes to check at next level
    while nextlevel:
        thislevel=nextlevel  # advance to next level
        nextlevel={}         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                seen[v]=level # set the level of vertex v
                if direction == 'forwards':
                    nextlevel.update(G[v]) # add neighbors of v
                elif direction == 'backwards':
                    nextlevel.update(dict((src,None) for src in G.predecessors(v))) # add predecessors of v
                else:
                    raise Exception("Direction {} not supported".format(direction))
        if (cutoff is not None and cutoff <= level):  break
        level=level+1
    return seen  # return all path lengths as dictionary
    
def locality_scan(g, node, order=0, direction='out'):
    """order =0, statistic is node degree"""
    if order==0:
        if direction == 'in':
            stat = g.in_degree(node)
        else:
            stat = g.out_degree(node)
    else:
        dir_dict = {'in':'forwards', 'out':'backwards'}
        N = single_source_shortest_path_length(g, node, cutoff=order, 
                                               direction=dir_dict[direction])
        stat = len(N)
    return stat

# right now, this is just node degree.
def locality_statistic_all(g, locality_statistic = 'scan', order=1, 
                           direction='out' # in/out/both
                           ):
    node_stats     = {}
    node_neighbors = {}
    for n in g.nodes_iter():
        successors        = g.successors(n)
        predecessors      = g.predecessors(n)
        if direction == 'in':
            node_neighbors[n] = predecessors
        elif direction == 'out':
            node_neighbors[n] = successors
        else:
            neighbors = set(successors)
            neighbors.add(predecessors)
            node_neighbors[n] = neighbors
        if locality_statistic == 'scan':
            #node_stats[n] = len(neighbors)
            if direction != 'both':
                node_stats[n] = locality_scan(g, n, order=order, 
                                              direction=direction)
            else:
                stat_in = locality_scan(g, n, order=order, 
                                          direction='in')
                stat_out = locality_scan(g, n, order=order, 
                                          direction='out')
                node_stats[n] = max(stat_in, stat_out)
            #neighbors = set(successors)
            #neighbors.add(predecessors)
        elif locality_statistic == 'size':
            if direction != 'both':
                N = single_source_shortest_path_length(g, n, cutoff=order, 
                                                       direction=direction)
                N = N.keys()
            else:
                N_in  = single_source_shortest_path_length(g, n, cutoff=order, 
                                                       direction='backwards')
                N_out = single_source_shortest_path_length(g, n, cutoff=order, 
                                                       direction='forwards')
                N = set(N_in.keys())
                N.update(N_out.keys())
            #g_i = nx.subgraph(neighbors)
            g_i = nx.subgraph(N)
            node_stats[n] = g_i.number_of_edges()
        #g_i = nx.subgraph(neighbors)
    return {'g':g, 
     'node_stats':node_stats, 
     'node_neighbors':node_neighbors
     }
    
def scan_statistic_all(node_stats, node_neighbors):
    scan = {}
    for node, neighbors in node_neighbors.iteritems():
        scan[node] = 0
        if neighbors:
            scan[node] = max([node_stats[n] for n in neighbors])
    return scan

def df_from_dict_of_dicts(all_scan, base_df):
    if base_df is None:
        df = pd.DataFrame()
    else:
        df = base_df
    # coerce 'all_scan' items to dataframes
    for date, scan in all_scan.iteritems():
        df_d = pd.DataFrame({date:scan})
        df = pd.concat([df, df_d], axis=1)
    df = df.fillna(0)
    df.sort_index(axis=1, inplace=True)
    return df
    
def normalize_stat(all_scan, k=5, additive_smoothing=0, base_df=None):
    """
    if base_df is None:
        df = pd.DataFrame()
    else:
        df = base_df
    # coerce 'all_scan' items to dataframes
    for date, scan in all_scan.iteritems():
        df_d = pd.DataFrame({date:scan})
        df = pd.concat([df, df_d], axis=1)
    df = df.fillna(0)
    """
    df = df_from_dict_of_dicts(all_scan, base_df)
    df = df + additive_smoothing
    n = df.shape[1]
    n0 = 0
    n1 = k
    mu = {}
    sigma = {}
    while n1 <= n:
        ix = int(np.mean([n0, n1-1]))
        col = df.columns[ix]
        mu[col]    = df.iloc[:, n0:n1].mean(axis=1)
        s = df.iloc[:, n0:n1].std(axis=1)
        df_s = pd.DataFrame({col:s, 'v0':np.ones(len(s))})
        #sigma[col] = np.vstack([s.values, np.ones(len(s))]).max(axis=1)
        sigma[col] = df_s.max(axis=1)
        n0+=1
        n1+=1
    df_mu = pd.DataFrame(mu)
    df_s  = pd.DataFrame(sigma)
    #df2 = df - df_mu
    #df3 = df2/df_s
    #return {'mu':df_mu, 'sigma':df_s, 'df':df, 'df2':df2, 'df3':df3}
    df = df - df_mu
    df = df/df_s
    df = df.dropna(axis=1)
    return df
    
########################################################
    
def naive_graph_edit_distance_adj(adj1, adj2):
    adj1[adj1>0] = 1
    adj2[adj2>0] = 1
    return np.abs(adj1-adj2).sum()
    
def naive_graph_edit_distance(g1, g2):
    nodes = set(g1.nodes())
    nodes.update(g2.nodes())
    nodes = list(nodes)
    adj1 = nx.adjacency_matrix(g1, nodelist=nodes)
    adj2 = nx.adjacency_matrix(g2, nodelist=nodes)
    return naive_graph_edit_distance_adj(adj1, adj2)
    
def jaccard_coef(a,b):
    a_s = set(a)
    b_s = set(b)
    return len(a_s.intersection(b_s)) / len(a_s.union(b_s))
    
def graph_jaccard(g1, g2):
    return jaccard_coef(g1.edges(), g2.edges())
    
def mean_graph_jaccard(graphs, target_ix=-1, return_adj = False, adj=None, nodes=None): # appears to be slower than GED alternative
    """
    Given a list of graphs, returns a graph defined by the set of all edges
    that appear in at least half of the input graphs
    """
    # Build nodelist
    if nodes is None:
        nodes = build_nodelist(graphs)
    
    # Count edge expectations
    adj = construct_mean_graph_counter(graphs, nodes=nodes)
    
    # Get edit distance
    g_target = graphs[target_ix]
    #adj_target = nx.adjacency_matrix(g_target, nodelist=nodes)
    adj_target = Counter(g_target.edges())
    #retval = {'ged':naive_graph_edit_distance_adj(adj, adj_target)}
    retval = {'ged':jaccard_coef(adj.keys(), adj_target.keys())}
    if return_adj:
        retval['adj_mean_graph'] = adj
    return retval

def build_nodelist(graphs):
    nodes = set()
    for g in graphs:
        nodes.update(g.nodes())
    nodes = list(nodes)
    return nodes
    
def construct_mean_graph(graphs, as_adjacency=True, nodes=None, format='csr'):
    if nodes is None:
        nodes = build_nodelist(graphs)
    
    # Count edge expectations
    adj=None
    for g in graphs:
        #g_adj = nx.adjacency_matrix(g, nodelist=nodes)
        g_adj = nx.to_scipy_sparse_matrix(g, nodelist=nodes, format=format)
        if adj is not None:
            adj += g_adj
        else:
            adj = g_adj
    adj = binarize(adj, threshold=len(graphs)/2)
    retval = adj
    
    if not as_adjacency:
        g = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        nodes_map = dict((ix, name) for ix, name in enumerate(nodes))
        g = nx.relabel_nodes(g, nodes_map)
        retval = g
    return retval
    
def construct_mean_graph_counter(graphs, as_adjacency=True, nodes=None):
    if nodes is None:
        nodes = build_nodelist(graphs)
    
    # Count edge expectations
    adj=None
    for g in graphs:
        g_adj = Counter(g.edges())
        if adj is not None:
            adj += g_adj
        else:
            adj = g_adj
    ### The motivation of the following code block is to eliminate calling nx.adjacency_matrix 
    # Filter low count edges
    # Via http://stackoverflow.com/questions/15861739/removing-objects-whose-counts-are-less-than-threshold-in-counter
    threshold = len(graphs)/2
    for key, count in dropwhile(lambda key_count: key_count[1] >= threshold, adj.most_common()):
        del adj[key]
    
    retval = adj
    
    if not as_adjacency:
        g = nx.DiGraph()
        g.add_edges_from([ (k[0], k[1], {'weight':v}) for k,v in adj.iteritems() ])
        retval = g
    
    return retval
    
def mean_graph_ged(graphs, target_ix=-1, return_adj = False, adj=None):
    """
    Given a list of graphs, returns a graph defined by the set of all edges
    that appear in at least half of the input graphs
    """
    # Build nodelist
    nodes = set()
    for g in graphs:
        nodes.update(g.nodes())
    nodes = list(nodes)
    
    start = time.time()
    if adj is None:
        adj = construct_mean_graph(graphs, nodes=nodes)
    end = time.time()
    print "mean graph built:", end-start

    # Get edit distance
    g_target = graphs[target_ix]
    adj_target = nx.adjacency_matrix(g_target, nodelist=nodes)
    retval = {'ged':naive_graph_edit_distance_adj(adj, adj_target)}
    if return_adj:
        retval['adj_mean_graph'] = adj
    return retval
    
def mean_graph_ged_ratio(graphs, target_ix=-1, target_is_denominator=False, default=0):
    """
    Returns the ratio of the GED between the target graph and the mean graph
    divided by the size (|E|) of either the target graph or mean graph, 
    depending on the value of the `target_is_denominator` parameter.
    
    graphs: A list of networkx graph objects
    target_ix: The index giving the position in the `graphs` list of the graph
        to be compared against the mean graph for calculating graph edit dist.
    target_is_denominator: whether the `target` graph should be used for 
        calculating the denominator of the ration (|E|) or whether the mean
        graph should be used.
    default: If denominator is zero, return this value.
    """
    ged = mean_graph_ged(graphs, target_ix, return_adj = not target_is_denominator)
    if target_is_denominator:
        denom = len(graphs[target_ix])
    else:
        adj = ged['adj_mean_graph']
        g_adj = nx.from_scipy_sparse_matrix(adj)
        denom = len(g_adj)
    retval = default
    if denom > 0:
        retval = ged['ged']/denom
    return retval
        
def directed_ego_graph(g, n, radius=1):
    """Ego graph where radius is only relative to successors of root node"""
    if radius == 1:
        nodes = g.neighbors(n)
        nodes.append(n)
        return nx.subgraph(g, nodes) # Not sure this saves any time relative to just running everything using the code in the 'else' clause
    else:
        distance = {n:0} # we'll track what we've visited here.
        for parent, child in nx.bfs_edges(g, n):
            d_parent = distance[parent]
            if d_parent > radius - 1:
                break
            if child not in distance:
                distance[child] = d_parent + 1
        return nx.subgraph(g, distance.keys())
        
def ged_all(graphs, radius = 1, target_ix=-1, score_as_ratio=True, target_is_denominator=False, verbose=False):
    """
    Calculates the graph edit distance (ratio) for all nodes in the target graph
    with respect to the mean graph of the input list of graphs.
    """
    nodes = graphs[target_ix].nodes()
    score = {}
    for n in nodes: ##### This loop is painfully slow.
        if verbose: print n
        #subgraphs = [nx.ego_graph(g, n, radius) for g in graphs]
        subgraphs = []
        for g in graphs:
            if g.has_node(n):
                #sub_g = nx.ego_graph(g, n, radius)
                sub_g = directed_ego_graph(g, n, radius)
            else:
                sub_g = nx.Graph()
            subgraphs.append(sub_g) # Overlapping subproblems for separate ged_all calls!!!
        if score_as_ratio:
            score[n] = mean_graph_ged_ratio(graphs, target_ix,  
                            target_is_denominator=target_is_denominator)
        else:
            score[n] = mean_graph_ged(graphs, target_ix)
    return score
        
# Via http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in xrange(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win
        
def window_apply(seq, n, func, *args, **kargs):
    retval = []
    for i,e in enumerate(window(seq, n)):
        print "window_apply iteration", i
        retval.append(func(e, *args, **kargs))
    return retval
    
        
if __name__ == '__main__':
    conn = sqlite3.connect(r"../subreddit_mentions.db")
    t_start = conn.execute('select min(created_utc) from mentions').fetchone()[0]
    t_end   = conn.execute('select max(created_utc) from mentions').fetchone()[0]
    graphs = get_time_graphs_range(conn, 
                                   t_start, 
                                   t_end, 
                                   interval_length = 7 * (60*60*24),
                                   delta           = 7 * (60*60*24),
                                   verbose=True
                                   )
    ## Wrap this block in a function
    # all_scan = {}
    # for g_t in graphs:
        # t0, t1, g = g_t
        # print t0, t1
        # local = locality_statistic_all(g, direction='in')
        # scan = scan_statistic_all(local['node_stats'], local['node_neighbors'])
        # all_scan[t0] = scan
    
    # nodes = get_nodes_list(conn)
    # base_df = pd.DataFrame({'dummy':pd.Series(np.nan,index=nodes)})
    # normalized = normalize_stat(all_scan, k=15, base_df=base_df, additive_smoothing=10)
    # #normalized[normalized.index=='askreddit'].T.abs().plot()
    
    # scan_max = normalized.max(axis=0) # This isn't very promising...
    # #scan_max.plot()
    

    ###########################################3

    # Alternative approach:
    # For a given time slice, filter the graph down to only significant edges.
    # Then disregard edge weight and calculate the graph edit distance between
    # each node's neighborhood (induced subgraph for some arbitrary depth) and
    # the median graph for the neighborhood. We can then calculate the locality 
    # statistic as something like the ratio between the edit distance and the size
    # of the neighborhood for the median graph (nodes? Edges? Nodes + edges? Just 
    # is probably fine. Not sure we need to count adding an edge to a non-existent
    # node as two separate operations. Might be worth trying.)
        
    #filtered_graphs = [filter_graph(g, alpha=.1, return_filtered_copy=True)['filtered_graph'] for g in graphs]
    filtered_graphs = []
    for g_t in graphs:
        t0, t1, g = g_t
        fg = filter_graph(g, alpha=.1, return_filtered_copy=True)['filtered_graph']
        filtered_graphs.append(fg)
        print "{} {} | ({} {}) -> ({} {})".format(t0,t1, len(g), len(g.edges()), len(fg), len(fg.edges()))
    
    #n_graphs=5
    #results = window_apply(filtered_graphs, n_graphs, ged_all)
    #test = ged_all(filtered_graphs[:5], verbose=True)