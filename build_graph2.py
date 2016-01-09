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

gf = filter_graph(g, .01)

print "Filtering inverted graph..."

gf_inv = filter_graph(g_inv, .01)

print "Writing filtered graph to file..."

nx.write_gexf(gf, "subreddit_graph_filtered.gexf")

print "Writing inverted filtered graph to file..."

nx.write_gexf(gf_inv, "subreddit_graph_inverted_filtered.gexf")