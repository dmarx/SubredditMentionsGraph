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
    import json

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
        graphs.append({'graph':g, 'start':start, 'end':end})
        start = start + increment
        end = start + window
    return graphs
    
def collapse_graph_snapshots(graphs):
    """
    Takes a list of graphs (output from get_graphs_in_range) and returns a 
    single json object where each edge has a "snapshots" attribute that gives
    tuples of (timestamp, weight) for each snapshot during which the edge was
    live.
    """
    edges, nodes, nodeslist = {}, [], []
    for g_dict in graphs:
        g = g_dict['graph']
        g['edges']['e_str'] = g['edges']['source'] + '~' + g['edges']['target']
        for _, e in g['edges'].iterrows():
            if not edges.has_key(e.e_str):
                edges[e.e_str] = {'source':e.source, 'target':e.target, 'snapshots':[]}
            edges[e.e_str]['snapshots'].append( (g_dict['end'], e.weight) )
        for n in g['nodes']:
            if n[0] not in nodeslist:
                nodeslist.append(n[0])
                nodes.append(n[1])
    return {'nodes':nodes, 'edges':edges.values()}
    
def calculate_layout(graph):
    return nx.spring_layout(graph) # Change to Force Atlas later.
    
if __name__ == '__main__':
    start    = dt.datetime(2015,01,01)
    end      = dt.datetime.utcnow()
    window   = dt.timedelta(days=7)
    interval = dt.timedelta(days=7)
    fname    = 'dynamic_graph.json'
    
    graphs = get_graphs_in_range(start, end)
    j = collapse_graph_snapshots(graphs)
    
    # Calculate layout on collapsed graph (union of subgraphs), apply as node
    # attribute
    full = get_graph_snapshot(start, end)
    layout = calculate_layout(full['g'])
    
    
    with open(fname, 'wb') as f:
        f.write(json.dumps(j))