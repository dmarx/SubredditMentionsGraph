#cd E:\Projects\subreddit_mentions_graph
#C:\Users\davidmarx\Documents\Projects\Toy Projects\Subreddit_Mentions_Graph
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
#df.to_csv('subreddit_edges.scv') # filter bad nodes out below

######################################

# Fold in goldensights' subreddit metadata.
# https://github.com/voussoir/reddit/tree/master/SubredditBirthdays

conn_gs = sqlite3.connect(r"sql.db")
#gs_star = pd.read_sql("select * from subreddits", conn_gs)    
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

#4 submission types, 7 subreddit types. What is this?

## This doesn't include our vote thresholding.
#mentioned_nodes = pd.read_sql("""
#    SELECT DISTINCT LOWER(source_subr) subr FROM mentions 
#    UNION 
#    SELECT DISTINCT LOWER(target_subr) subr FROM mentions""", conn)

mentioned_nodes = pd.concat([df.source, df.target]).unique()

#nodes = nodelist.merge(gs, how="inner", left_on="subr", right_on="name_lwr")
#nodes = gs[gs["name_lwr"].isin(mentioned_nodes['subr'])]
nodes = gs[gs["name_lwr"].isin(mentioned_nodes)]

### Filter down edgelist to valid nodes
#df = df.merge(nodes[['name','name_lwr']], how="inner", left_on="source", right_on="name_lwr")
#df.source = df.name
#df = df[["source","target","weight"]].merge(nodes[['name','name_lwr']], how="inner", left_on="target", right_on="name_lwr")
#df.target = df.name
#df = df[["source","target","weight"]]
###df.to_csv('subreddit_edges.csv')

#df2 = df[df["source"].isin(nodes["name_lwr"])]
#df3 = df2[df2["target"].isin(nodes["name_lwr"])]


df = df[df["source"].isin(nodes["name_lwr"])]
df = df[df["target"].isin(nodes["name_lwr"])]

#nodes['id'] = nodes.name
#nodes[["id","created_date","nsfw","subscribers"]].to_csv('nodelist.csv')

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