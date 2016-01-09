import networkx as nx

def invert_graph(g):
    """
    Returns a copy of the input graph where the direction of all edges is 
    reversed.
    """
    source, target, weight = zip(*g.edges(data=True))
    inv_edges = zip(target, source, weight)
    g2 = nx.DiGraph()
    g2.add_nodes_from(g.nodes(data=True))
    g2.add_edges_from(inv_edges)
    return g2
        
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