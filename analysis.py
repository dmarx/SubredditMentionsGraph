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
        AND subscribers >= {}
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
                 'log_subscribers':int(np.floor(np.log(n.subscribers))) + 1
                 }) for _,n in nodes.iterrows()]
g=nx.DiGraph()
g.add_nodes_from(node_tuples)
g.add_weighted_edges_from(df.values)
nx.write_gexf(g, "subreddit_graph.gexf")

# 12795 Nodes
# 67068 edges

# NB: Nodelist doesn't include "blackout2015". Probably missing
# a lot of newer subs. :(

######################################

# 1. select author, source, target, min(date)
# 2. group by author, source, target
# 3. identify date associated with kth-record (for different fixed values of k) in group
# 4. draw curve giving cumulative sum of date counts (i.e. unique observed edges, thresholded at k votes)

df = pd.read_sql("""
    SELECT  author,
            LOWER(source_subr) source,
            LOWER(target_subr) target,
            min(created_utc) created_utc
    FROM    mentions
    WHERE   LOWER(source_subr) <> LOWER(target_subr)
    GROUP BY author, LOWER(source_subr), LOWER(target_subr)
    """.format(k=k), conn)

start = time.time()
k=3
grouped = df.groupby(['source','target'])
edge_date = []
edge_interval = []
node_date = {}
node_interval = {}
for src_tgt, grp in grouped:
    if grp.shape[0] < k:
        continue
        
    d0 = grp.iloc[0].created_utc
    dk = grp.iloc[k-1].created_utc
    delta = dk-d0
    edge_date.append(dk)
    edge_interval.append(delta)
    
    src, tgt = src_tgt
    if not node_date.has_key(src):
        node_date[src] = dk
        node_interval[src] = delta
    if not node_date.has_key(tgt):
        node_date[tgt] = dk
        node_interval[tgt] = delta
end = time.time()
print "Elapsed:", end-start # 101 seconds
print "Edges:", len(edge_date) # k=3, 66048 edges
print "Nodes:", len(node_date) # k=3, 14414

edge_date.sort()
s_edates = pd.Series(edge_date)
s_edates = (s_edates - s_edates[0])/(60*60*24) 
plt.plot(s_edates, range(len(s_edates))) # number of edges grows approximately linearly

node_dates = node_date.values()
node_dates.sort()
s_ndates = pd.Series(node_dates)
s_ndates = (s_ndates - s_ndates[0])/(60*60*24)
plt.plot(s_ndates, range(len(s_ndates))) # approximately linear growth of nodes as well. Should I add some sort of windowing? Ensure we're counting *active* subs?

# Distribution of arrival times of new nodes in graph for a given value of k
node_intervals = node_interval.values()
node_intervals.sort()
s_nintvl = pd.Series(node_intervals)
s_nintvl = (s_nintvl - s_nintvl[0])/(60*60*24)
s_nintvl.hist(bins=100)
# What is going on here? Why is there this insane spike at around 105-10 days? Appears regardless of k (tested with k=3,20)
# Is this maybe a function of a large time gap in my dataset separating the second and third occurence of lots of a nodes first edge...?