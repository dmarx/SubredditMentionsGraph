"
- number of nodes
- number of edges
- clustering coefficient
-  of communities
  - power law fit
- avg shortest path length
- modularity
- small-worldness score
"

calc_graph_stats = function(g, n_comm=FALSE){
  retval = list()
  
  # number of nodes
  retval['n_nodes'] = gorder(g)
  
  # number of edges
  retval['n_edges'] = ecount(g)
  
  # clustering coeefficient / power law fit (?)
  retval['transitivity'] = transitivity(g, type="global", isolates="NaN")
  
  if(n_comm){
    wtc = walktrap.community(g)
    retval['n_comm'] = length(unique(wtc$membership))
  }
  
  return(retval)
}

# Modifed from:
#   https://github.com/alessandrobessi/disparityfilter/blob/master/R/disparity_filter.R
calculate_edge_alpha <- function(G, weights = NA, mode = "all") {
  if(is.na(weights)){
    weights = E(g)$weight
  }
  d = degree(G, mode = mode)
  e = cbind(as_data_frame(G)[, 1:2 ], weight = weights, alpha = NA)
  if (mode == "all") {
    e = rbind(e, data.frame(from = e[, 2], to = e[, 1], e[, 3:4 ]))
  }
  
  for (u in which(d > 1)) {
    
    w = switch(substr(mode, 1, 1),
               a = which(e[, 1] == u | e[, 2] == u),
               i = which(e[, 2] == u),
               o = which(e[, 1] == u)
    )
    w = sum(e$weight[ w ]) / (1 + (mode == "all"))
    
    k = d[u]
    
    for (v in ego(G, 1, u, mode)[[1]][-1]) {
      
      ij = switch(substr(mode, 1, 1),
                  a = which(e[, 1] == u & e[, 2] == v),
                  i = which(e[, 1] == v & e[, 2] == u),
                  o = which(e[, 1] == u & e[, 2] == v)
      )
      # cat(mode, "-", u, "->", v, ":", ij, "\n")
      
      # p_ij = e$weight[ ij ] / w
      # alpha_ij = integrate(function(x) { (1 - x) ^ (k - 2) }, 0, p_ij)
      # alpha_ij = 1 - (k - 1) * alpha_ij$value
      e$alpha[ ij ] = (1 - e$weight[ ij ] / w) ^ (k - 1)
      
    }
    
  }
  
  #return(e[ !is.na(e$alpha) & e$alpha < alpha, 1:4 ])
  return( e[ !is.na(e$alpha), ] )
  
}

disparity_filter <-  function(alpha, g=NA, e=NA){
  if(is.na(e)){
    e = calculate_edge_alpha(g)
  }
  return(e[ !is.na(e$alpha) & e$alpha < alpha, 1:4 ])
}

# The code in the disparityfilter package is stupid slow. Let's try this other' guy's code. Modified from:
#   https://github.com/bobvdvelde/ICA15/blob/e86e8e4c1c944455b3af04f221e793d4e41098af/backbone.r
library(Matrix)

backbone <- function(g){
  mat = get.adjacency(g, attr='weight')
  if(!is.directed(g)) mat[lower.tri(mat)] = 0 # prevents counting edges double in symmetric matrix (undirected graph)
  weightsum = rowSums(mat) + colSums(mat)
  #k = rowSums(mat>0) + colSums(mat>0)
  k_out = rowSums(mat>0) 
  k_in  = colSums(mat>0)
  if(is.directed(g)){
    k_out = k_out + k_in
    k_in = k_out
  }
  
  edgelist_ids = get.edgelist(g, names=F)
  alpha_ij = getAlpha(mat, weightsum, k_out, edgelist_ids) # alpha for edges from i to j
  alpha_ji = getAlpha(mat, weightsum, k_in, edgelist_ids, transpose=T)
  alpha_ij[alpha_ji < alpha_ij] = alpha_ji[alpha_ji < alpha_ij]
  #alpha_ij
  set.edge.attribute(g, "alpha", index=edgelist_ids, alpha_ij)
}

getAlpha <- function(mat, weightsum, k, edgelist_ids, transpose=F){
  if(transpose) mat = t(mat)
  p_ij = mat / weightsum
  alpha = ((1 - p_ij[edgelist_ids])^(k[edgelist_ids[,1]]-1))
  if(transpose) alpha = t(alpha)
  alpha
}

if(FALSE){
  setwd("C:/Users/davidmarx/Documents/Projects/Toy Projects/Subreddit_Mentions_Graph")
  library(igraph)
  library(RSQLite)
  library(data.table)
  library(lubridate)
  
  db_path = "subreddit_mentions.db"
  conn = dbConnect(SQLite(), db_path)
  
  mentions = data.table(dbGetQuery(conn, "select * from mentions"))
  setnames(mentions, c("source_subr_lwr", "target_subr_lwr"), c("Source", "Target"))
  setkey(mentions, source_subr, target_subr)
  
  g_edgelist = mentions[,list(weight=length(unique(author_lwr))), by=list(source_subr, target_subr)]
  g = graph.data.frame(g_edgelist)
  
  #################################
  
  # These are super fast
  system.time( n_nodes <- gorder(g) ) # 0
  system.time( n_edges <- ecount(g) ) # 0
  
  system.time( trans <- transitivity(g, type="global") ) # 2.14 sec 
  system.time( avg_pathlen <- average.path.length(g) ) # 176.58 sec
  
  system.time( fgc <- fastgreedy.community(as.undirected(g)) ) # ...3:57 (-5min?)
  #system.time( ebc <- edge.betweenness.community(g) ) # Expect this to be slow :(
  #system.time( fgc <- fastgreedy.community(g) ) ## undirected graphs only
  #wtc <- walktrap.community(g) ## slow
  
  #########################
  
  system.time(test <- calc_graph_stats(g))
  
  #install.packages( 'disparityfilter'  )
  
  #system.time( e <- calculate_edge_alpha(g) ) # this, not surprisingly, takes a little while
  # This probably means that disparity filter package could be significantly improved. My python
  # implementation was blazingly fast compared to this.
  system.time( e <- backbone(g) ) # 3.08
  alpha = get.edge.attribute(e, "alpha")
  weight = get.edge.attribute(e, "weight")
  edges = get.edgelist(e)
  edgelist = data.table(cbind(edges, weight, alpha))
  names(edgelist)[1:2] = c("source", "target")
  setkey(edgelist, alpha)
  
  a = 10^seq(-4,0,length.out=30)
  #a = 10^seq(-2,0,length.out=30)
  a = a[-1]# Not sure why, but results are funny for a=1e-4
  a = a[-length(a)]
  build_filtered_graph= function(a, el=edgelist){
    graph.data.frame(el[alpha<=a])
  }
  #test = build_filtered_graph(.0001) # Not sure why, but results are funny for a=1e-4
  stats = sapply(a, function(x) calc_graph_stats( build_filtered_graph(x), n_comm=TRUE ))
  stats = data.table(t(stats))
  stats$alpha = a
  stats[,n_nodes := unlist(n_nodes)]
  stats[,n_edges := unlist(n_edges)]
  stats[,transitivity := unlist(transitivity)]
  stats[,n_comm := unlist(n_comm)]
  
  
  stats[, plot(alpha, n_nodes/gorder(g), log='x', ylim=c(0,.1))]
  stats[, plot(alpha, n_edges/ecount(g), log='x', ylim=c(0,.04))]
  stats[, plot(alpha, transitivity, log='x')]
  stats[, plot(alpha, n_comm, log='x', ylim=c(0, max(n_comm)))]
  
  g2 = build_filtered_graph(.001)
  wtc = walktrap.community(g2)
  names(wtc)
  length(unique(wtc$membership))#2381
}


Matrix(c(1,2,0,4,0,6,0,8,9), 3)^2
