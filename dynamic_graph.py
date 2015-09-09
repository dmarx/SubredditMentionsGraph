#cd E:\Projects\subreddit_mentions_graph
#cd C:\Users\davidmarx\Documents\Projects\Toy Projects\Subreddit_Mentions_Graph
import sqlite3
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import time
import datetime as dt
try:
    import ujson as json
except:
    import json # I don't think this can handle datetime objects. 
                # Neither handles numpy.
                # ... wait... shouldn't be datetime. Should coerce to epoch anyway.

conn = sqlite3.connect(r"subreddit_mentions.db")
conn_gs = sqlite3.connect(r"sql.db")

def dt_to_epoch(date, reference=dt.datetime(1970,1,1)):
    """Converts a """
    return (date-reference).total_seconds()
    
                         
def get_graph_snapshot(start, end, n_subscribers_threshold = 50):
    """
    Return the relevant subgraph within a set time range defiend by datetime 
    objects
    """

    df = pd.read_sql("""
        SELECT  LOWER(source_subr) source,
                LOWER(target_subr) target,
                COUNT(DISTINCT author) weight
        FROM    mentions
        WHERE   LOWER(source_subr) <> LOWER(target_subr)
        AND     created_utc between {start} and {end}
        GROUP BY LOWER(source_subr), LOWER(target_subr)
        HAVING COUNT(DISTINCT author) > 2
        """.format(start=dt_to_epoch(start), end=dt_to_epoch(end)), conn)

    # Fold in goldensights' subreddit metadata.
    # https://github.com/voussoir/reddit/tree/master/SubredditBirthdays

    gs = pd.read_sql("""
        SELECT 
            idstr,
            created created_epoch,
            human created_date,
            name,
            LOWER(name) name_lwr,
            nsfw,
            subscribers,
            subreddit_type,
            submission_type
        FROM subreddits
        WHERE 1=1
            AND (subscribers >= {}
            OR subreddit_type in (2,3,6) --private, archived, gold
            --OR lower(name) in ('jailbait', 'fatpeoplehate', 'hamplanethatred', 'transfags', 'neofag', 'shitniggersay', 'thefappening')
            )
        """.format(n_subscribers_threshold), conn_gs)

    ### Filter down edgelist to valid nodes
    mentioned_nodes = pd.concat([df.source, df.target]).unique()
    nodes = gs[gs["name_lwr"].isin(mentioned_nodes)]
    df = df[df["source"].isin(nodes["name_lwr"])]
    df = df[df["target"].isin(nodes["name_lwr"])]

    # Construct an gexf dump of the graph to facilitate (hopefully?) importing to gephi

    node_tuples = [(n['name_lwr'], 
                    {'Name':n['name'],
                     'Created Date':n.created_date, 
                     'URL':"http://www.reddit.com/r/{}".format(n['name']), 
                     'NSFW':n.nsfw==1, 
                     'Number of Subscribers':n.subscribers,
                     'log_subscribers':int(np.floor(np.log(n.subscribers + 1))) + 1
                     }) for _,n in nodes.iterrows()]
    g=nx.DiGraph()
    g.add_nodes_from(node_tuples)
    g.add_weighted_edges_from(df.values)
    #nx.write_gexf(g, "subreddit_graph.gexf")
    return {'edges':df, 'nodes':node_tuples, 'g':g}
    
def get_graphs_in_range(start, 
                        end, 
                        window=dt.timedelta(days=7),
                        increment=dt.timedelta(days=7),
                        n_subscribers_threshold = 0
                        ):
    graphs = []
    low,high = start, end
    end = start + window
    while end <= high:
        g = get_graph_snapshot(start, end, n_subscribers_threshold)
        graphs.append({'graph':g, 'start':dt_to_epoch(start), 'end':dt_to_epoch(end)})
        start = start + increment
        end = start + window
    return graphs
    
def collapse_graph_snapshots(graphs, layout=None):
    """
    Takes a list of graphs (output from get_graphs_in_range) and returns a 
    single json object where each edge has a "snapshots" attribute that gives
    tuples of (timestamp, weight) for each snapshot during which the edge was
    live.
    """
    edges, nodes, nodeslist = {}, {}, []
    for g_dict in graphs:
        g = g_dict['graph']
        g['edges']['e_str'] = g['edges']['source'] + '~' + g['edges']['target']
        for _, e in g['edges'].iterrows():
            if not edges.has_key(e.e_str):
                edges[e.e_str] = {'source':e.source, 'target':e.target, 'snapshots':[]}
            edges[e.e_str]['snapshots'].append( (g_dict['end'], e.weight) )
        for n in g['nodes']:
            #if n[0] not in nodeslist:
                #nodeslist.append(n[0])
                #nodes.append(n[1])
            if not nodes.has_key(n[0]):
                nodes[n[0]] = n[1]
    if layout:
        for n, xy in layout.iteritems():
            if nodes.has_key(n):
                nodes[n]['cx'] = float(xy[0])
                nodes[n]['cy'] = float(xy[1])
        for n in nodes.keys():
            if not layout.has_key(n):
                nodes.pop(n)
    return {'nodes':nodes.values(), 'edges':edges.values()}
    
def calculate_layout(graph):
    # This would probably work better (even as a spring layout) using the log
    # of edge weight instead of the actual edge weight.
    #return nx.spring_layout(graph) # Change to Force Atlas later.
    return forceatlas2_layout(graph, linlog=True, nohubs=False, iterations=100)
    
################################################################################

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

def matrix_index_to_condensed_vector(i,j,n):
    return i*(n-1) + j    
    
def condensed_vector_to_matrix_index(ix, n):
    j = ix % n
    i =  (ix-j )/n
    return i, j
    
def edge_to_condensed_vector_index(source, target, nodelist):
    """
    Converts a single edge in i->j notation to its condensed vector index
    relative to an input nodelist.
    """
    i = nodelist.index(source)
    j = nodelist.index(target)
    n = len(nodelist)
    return matrix_index_to_condensed_vector(i,j,n)
        
def graph_to_condensed_vector(g, nodelist):
    """
    Given an input networkx graph and nodelist, returns the condensed vector
    form of the graph relative to the given nodelist.
    """
    n = len(nodelist)
    v = np.zeros(n**2)
    for source, target, attributes in g.edges_iter(data=True):
        ix = edge_to_condensed_vector_index(source, target, nodelist)
        v[ix] = attributes['weight']
    return v
            
def dynamic_graph_to_condensed_vector(graph, nodelist=None):
    """
    Given an input dynamic graph object (nx.DiGraph, start, end) returns the 
    adjacency list converted to a condensed vector form.
    """
    graph = graph['graph']
    if not nodelist:
        nodelist = [n[0] for n in graph['nodes']]
    return graph_to_condensed_vector(graph['g'], nodelist)
            
def construct_nodelist(graphs):
    """
    Given an input list of dynamic graph objects, returns the union of nodes
    on each graph.
    """
    nodes = set()
    for d_graph in graphs:
        graph = d_graph['graph']
        #print graph
        #print graph['nodes']
        g_nodes = [n for n,_ in graph['nodes']]
        nodes.update(g_nodes)
    nodes = list(nodes)
    nodes.sort()
    return nodes
            
def flatten_graphs_into_df(graphs):
    """
    Takes a list of graphs (as outputted by get_graphs_in_range) and returns
    a dataframe where each graphs is a... column?
    
    Let's break it down: what do I NEED here? I want to be able to do two things:
        1. For a given edge, determine the distribution of its weight over time.
        2. For a given neighborhood, determine the edit distance between the 
           neighborhood and the median/mean graph for the neighorhbood.
    
    Option 2 would be easy enough to encode ina  flat dataframe. Option 1 would be a bit harder.
    For option 2, I just have a vector for the node list. For option 1 I'd need to flatten the entire
    full adjacency matrix into a vector. Which... I guess is not impossible. And I could do that now
    without building the infrastructure for the graph edit distance calculations. 
    
    ...OK, let's flatten adjacency matrices into column vectors.
    
    1. Take the nodelist as input and assign each node an index.
    2. Convert an edge to its node index notation i->j.
    3. Use condensed distance matrix tools from TAD to determine the flattened
       index corresponding to the undirected edge i->j
       --> is this acceptable? I think I want to keep the edges directed. 
           Honestly, it probably doesn't make a huge difference.
    
    """
    nodelist = construct_nodelist(graphs)
    df = pd.DataFrame()
    for graph in graphs:
        v = dynamic_graph_to_condensed_vector(graph, nodelist) # Really need to pass in a fixed nodelist.
        df_d = pd.DataFrame({graph['end']:v})
        df = pd.concat([df, df_d], axis=1)
    df = df.fillna(0)
    df.sort_index(axis=1, inplace=True)
    df['node'] = nodelist # need to construct this
    df.set_index(['node'])
    return df
        
        
################################################################################
    
if __name__ == '__main__':
    start    = dt.datetime(2015,01,01)
    end      = dt.datetime.utcnow()
    window   = dt.timedelta(days=7)
    interval = dt.timedelta(days=7)
    fname    = 'dynamic_graph.json'
    
    # Calculate layout on collapsed graph (union of subgraphs), apply as node
    # attribute
    full = get_graph_snapshot(start, end)
    ###############
    # Let's see what happens if we apply a logarithmic transformation to the edge weights
    #full['edges']['weight'] = np.log(full['edges']['weight'])
    #g=nx.DiGraph()
    #g.add_nodes_from(full['nodes'])
    #g.add_weighted_edges_from(full['edges'].values)
    #layout = calculate_layout(g)    
    #graphs = [{'graph':{'g':g, 'nodes':full['nodes'], 'edges':full['edges']}, 'start':dt_to_epoch(start), 'end':dt_to_epoch(end)}]
    ###############
    layout = calculate_layout(full['g'])
    
    graphs = get_graphs_in_range(start, end, window, interval)
    j = collapse_graph_snapshots(graphs, layout)
    
    with open(fname, 'wb') as f:
        f.write(json.dumps(j))