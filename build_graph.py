#cd E:\Projects\subreddit_mentions_graph
#cd C:\Users\davidmarx\Documents\Projects\Toy Projects\Subreddit_Mentions_Graph
import sqlite3
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

conn = sqlite3.connect(r"subreddit_mentions.db")
df = pd.read_sql("""
    SELECT  LOWER(source_subr) source,
            LOWER(target_subr) target,
            COUNT(DISTINCT author) weight
    FROM    mentions
    WHERE   LOWER(source_subr) <> LOWER(target_subr)
    GROUP BY LOWER(source_subr), LOWER(target_subr)
    HAVING COUNT(DISTINCT author) > 2
    """, conn)

# Fold in goldensights' subreddit metadata.
# https://github.com/voussoir/reddit/tree/master/SubredditBirthdays

conn_gs = sqlite3.connect(r"sql.db")
n_subscribers_threshold = 50
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
        OR lower(name) in ('jailbait', 'fatpeoplehate', 'hamplanethatred', 
        'transfags', 'neofag', 'shitniggersay', 'thefappening')
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
nx.write_gexf(g, "subreddit_graph.gexf")