"""
Instead of thresholding on subreddit size, let's follow rhiever's advice and see 
what the result looks like when we use the disparity filter instead.
"""
print "Importing libraries..."

import sqlite3
import pandas as pd
import networkx as nx
import numpy as np
from time_series_anomaly_detection.edge_significance import filter_graph
from time_series_anomaly_detection.graph_utilities import invert_graph

print "Calculating edges..."

conn = sqlite3.connect(r"subreddit_mentions.db")
df = pd.read_sql("""
    SELECT  LOWER(source_subr) source,
            LOWER(target_subr) target,
            COUNT(DISTINCT author) weight
    FROM    mentions
    WHERE   LOWER(source_subr) <> LOWER(target_subr)
    GROUP BY LOWER(source_subr), LOWER(target_subr)
    """, conn)
    
print "Reading node metadata..."
    
# Add subreddit metadata from /u/GoldenSights' dataset
conn_gs = sqlite3.connect(r"sql.db")
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
    """, conn_gs)

print "Filtering nodes to relevant list..."
    
mentioned_nodes = pd.concat([df.source, df.target]).unique()
nodes = gs[gs["name_lwr"].isin(mentioned_nodes)]
df = df[df["source"].isin(nodes["name_lwr"])]
df = df[df["target"].isin(nodes["name_lwr"])]

# Construct an gexf dump of the graph to facilitate (hopefully?) importing to gephi

print "Adding metadata to node list..."

node_tuples = [(n['name_lwr'], 
                {'Name':n['name'],
                 'Created Date':n.created_date, 
                 'URL':"http://www.reddit.com/r/{}".format(n['name']), 
                 'NSFW':n.nsfw==1, 
                 'Number of Subscribers':n.subscribers,
                 'log_subscribers':int(np.floor(np.log(np.max([n.subscribers,1])))) + 1
                 }) for _,n in nodes.iterrows()]
                 
print "Building graph..."
                 
g=nx.DiGraph()
g.add_nodes_from(node_tuples)
g.add_weighted_edges_from(df.values)

print "Writing graph to file..."

nx.write_gexf(g, "subreddit_graph_unfiltered.gexf")

print "Inverting graph..."

g_inv = invert_graph(g)

print "Filtering graph..."

gf = filter_graph(g, .01, return_filtered_copy=True)

print "Filtering inverted graph..."

gf_inv = filter_graph(g_inv, .01, return_filtered_copy=True)

print "Re-inverting graph to original edge directions post-filtering..."

gf_inv['graph']          = invert_graph(gf_inv['graph'])
gf_inv['filtered_graph'] = invert_graph(gf_inv['filtered_graph'])

print "Writing filtered graph to file..."

nx.write_gexf(gf['graph'], "subreddit_graph_filtered.gexf")
nx.write_gexf(gf['filtered_graph'], "subreddit_graph_filtered_a01.gexf")

print "Writing inverted filtered graph to file..."

nx.write_gexf(gf_inv['graph'], "subreddit_graph_inverted_filtered.gexf")
nx.write_gexf(gf_inv['filtered_graph'], "subreddit_graph_inverted_filtered_a01.gexf")

print "Take union of both filtering approaches..."

#g_union = nx.union(gf['filtered_graph'], gf_inv['filtered_graph']) # This is stupid.
nodes_union = set( gf['filtered_graph'].nodes() )
nodes_union.update( gf_inv['filtered_graph'].nodes() )

edges_dict = {}
for u, v, d in gf['filtered_graph'].edges_iter(data=True):
    key = u + '|' + v
    edges_dict[key] = d
for u, v, d1 in gf_inv['filtered_graph'].edges_iter(data=True):
    key = u + '|' + v
    if edges_dict.has_key(key):
        d2 = edges_dict[key]
        d_new = {}
        for k in d2.keys():
            d_new[k] = max(d1[k], d2[k])
            edges_dict[key] = d_new
    else:
        edges_dict[key] = d1

g_union = nx.DiGraph()
g_union.add_nodes_from(nodes_union)
for k,v in edges_dict.iteritems():
    source, target = k.split('|')
    g_union.add_edges_from([[source, target,v]])

print "Persist union..."

nx.write_gexf(g_union, 'subreddit_graph_filtered_union.gexf')


# To do: I should really compare these graphs to the original heuristic based threshold filtering.