library(igraph)
#library(RSQLite)
library(data.table)
library(lubridate)

#####################################
# Define some convenience functions #
#####################################

epochToDate = function(val){as.POSIXct(val, origin="1970-01-01")}

get_peak_details = function(peak_ix){
  peak_date = d_seq[-1][peak_ix]
  node_id = test$arg_max_v[peak_ix]
  node = V(graphs[[peak_ix]])[node_id] # user_history_bot ... wtf? Weirdness.
  list(date=peak_date, node=node$name)
}

############################################
# Read in dynamic graph data and construct # 
# projections along regular time intervals #
############################################

#db_path = "subreddit_mentions.db"
#conn = dbConnect(SQLite(), db_path)

#mentions = data.table(dbGetQuery(conn, "select * from mentions"))
#setnames(mentions, c("source_subr_lwr", "target_subr_lwr"), c("Source", "Target"))


#####################################

#setwd("E:/Projects/SubredditMentionsGraph/")
bq_path = "mentions_from_bigquery"

mentions_bq <- lapply(list.files(bq_path, pattern="*.csv", full.names=TRUE), fread)
mentions_bq = rbindlist(mentions_bq)
mentions_bq[,.N] - mentions_bq[author!='[deleted]',.N] # 591345
setnames(mentions_bq, c("source","target"), c("Source","Target"))

mentions = mentions_bq
mentions[,date:=epochToDate(created_utc)]

save(mentions, file="rdata/mentions.rdata")

#####################################

setkey(mentions, created_utc)

nodelist = mentions[,union(Source, Target)]

#drange = mentions[,list(d_min=min(created_utc), d_max=max(created_utc))]
#drange = lapply(drange, epochToDate)
#drange$d_min = floor_date(drange$d_min, unit="day") + days(1)
#drange$d_max = floor_date(drange$d_max, unit="day")

#drange = list(d_min=ymd("2014-09-01"), d_max=ymd("2015-12-31"))
drange = list(d_min=ymd("2013-09-01"), d_max=ymd("2015-12-31"))

d_seq = seq(from=drange$d_min, to=drange$d_max, by="week")
#d_seq = seq(from=drange$d_min, to=drange$d_max, by="3 days")
d_seq_numeric = as.numeric(d_seq)

start = Sys.time()
graphs=list()
for(i in 2:length(d_seq)){
  d1 = as.numeric(d_seq[i-1])
  d2 = as.numeric(d_seq[i])
  print(c(i,d2))
  
  records = mentions[d1<created_utc & created_utc<=d2][,.(Source, Target, author)]
  setkey(records, Source, Target)
  
  edges = records[,list(Weight=length(unique(author))), by=list(Source, Target)]
  
  # Threshold edges to handle bots. Could also try disparity filter
  edges = edges[Weight>=2]
  
  g = graph.data.frame(edges, directed=TRUE, vertices=nodelist)
  graphs[as.character(d2)]=list(g)
}
end = Sys.time()
elaps = end - start
print(elaps) # 2.8 min to build 43 subgraphs
# 17min to build weekly graphs (70) going back to 2014-09

save(graphs, file="rdata/graphs.rdata")

###################################################
# Calculate scan statistics for anomaly detection #
###################################################

start = Sys.time()
test = scan_stat(graphs, tau=10) # Each graph is 1 week, so this is approx 2 months
end = Sys.time()
elaps = end - start
print(elaps) # 7min
#17min for 70 graphs with tau=10, k=1

save(test, file="rdata/test.rdata")

#########################
# Investigate anomalies #
#########################

plot(d_seq[-1], test$stat, type='l') # oh baby!
abline(v=ymd("2015-06-10"), col="red", lty=2) # FPH banned
abline(v=ymd("2015-07-02"), col='purple', lty=2) # Victoria fired / 2015 Blackout
#abline(v=ymd("2015-07-10"), col='blue', lty=2)  # Ellen Pao steps down, Huffman appointed
#abline(v=ymd("2015-11-20"), col='green', lty=2)  # Privacy policy changes announced
#abline(v=ymd("2015-08-13"), col='blue', lty=2)  # Reddit bans /r/watchpeopledie in germany
abline(v=ymd("2015-08-05"), col='blue', lty=2)  # Reddit bans /r/coontown and other racist subs https://www.reddit.com/r/announcements/comments/3fx2au/content_policy_update/ctsqobs ## I'm not convinced this is it.
abline(v=ymd("2015-04-01"), col='green', lty=2) # The button (april fools)


# Identify the most chaotic dates (weeks) and neighborhoods
details = data.table(t(sapply(which(!is.na(test$stat)), get_peak_details)))
details$stat = test$stat[!is.na(test$stat)]
details[order(stat, decreasing=TRUE)]

#################################
# Investigate population growth #
#################################

# Daily count of unique users making subreddit mentioning comments as a proxy
# for quantifying the size of the active userbase overtime (i.e. is the community shrinking?)
#active_users = mentions[,list(count=length(unique(author))), by=list(year, month, day, wday)]
active_users = mentions[,list(count=length(unique(author))), by=list(year(date), month(date), day(date))]
active_users[,date:=ymd(paste(year, month, day, sep='-'))]
active_users$index = 1:nrow(active_users)-1
setkey(active_users, date)
active_users[date>=ymd('2013-01-01')][,plot(date, count, type='l')] # oh shit... definitely shrinking, but plateauing. Looks like it's actually shrunk by about half
#mod = lm(count~index, active_users[date>=ymd('2015-01-01')])
#mod2 = lm(count~date, active_users[date>=ymd('2015-01-01')]) # just for display purposes
#abline(mod2, lty=2, col='blue')
#summary(mod)
#summary(mod)
#coef(mod) 

# For the last year, Reddit has been shrinking by about 18 (subreddit mentioning) people a day.
# It's not clear whether or not this is a true shrinking of the community or perhaps
# just them getting better at fighting bots and spam, but I strongly suspect it is in
# fact the former. Would be great if my dataset went further back.

# To show that this is an accurate proxy, I'd need to pull down the number of unique coment
# authors day-over-day (which would really be a better way of doing this anyway). I'm just
# counting people who mention subreddits. The size of the shirnkage is almost certainly more
# than 18 people a day, just as the active user base is absolutely bigger than 5-10K people.

# I should still compare this trend to the global active users to demonstrate that the
# arrival rate of this particular class of comments is reflective of global user activity.


pre_blackout  = active_users[date < ymd("2015-06-01") & date>=ymd('2015-01-01')]
post_blackout = active_users[date > ymd("2015-07-18")]

mod_pre = lm(log(count)~index, pre_blackout)
mod_post = lm(log(count)~index, post_blackout)

mod_pre_d = lm(log(count)~date, pre_blackout)
mod_post_D = lm(log(count)~date, post_blackout)
summary(mod_pre)
summary(mod_post) # reddit was already shrinking, but accelerated by an order of magnitude after the blackout
# current shriknage rate = 0.12%
with(active_users[date>=ymd('2015-01-01')],
  plot(date, count,
       xlab="Date", ylab="#Unique 'subreddit mentioning' users by day",
       main="Effect of the 2015 Blackout on the\nmagnitude of the active reddit user base",
       type='l'
       )
)
lines(pre_blackout$date, exp(predict(mod_pre_d)), lty=2, col='blue')
lines(post_blackout$date, exp(predict(mod_post)), lty=2, col='red')

coef(mod_pre)  # +0.0003104884 (+.03% change day over day)
coef(mod_post) # -0.0009321799 (-.09% change day over day)

coef(mod_post)/coef(mod_pre) 
# Reddit is currently shrinking 3x faster than it was growing prior to the 
# blackout

#############################################################
# Investigate community growth (focusing on giant component) #
#############################################################

# pretty sure this returns the size of the giant component
count_connected_nodes =  function(g){length(component_distribution(g))} 
n_nodes = sapply(graphs, count_connected_nodes)
#plot(d_seq[-1], n_nodes)
# The giant component is shrinking. 
# Let's do the same left-right analysis we did for active users

pre_ix  = d_seq < ymd("2015-06-01") & d_seq >= ymd("2015-01-01")
post_ix = d_seq > ymd("2015-07-04") & d_seq < ymd("2015-12-31")
n_nodes_pre  = n_nodes[pre_ix]
n_nodes_post = n_nodes[post_ix]

mod_comp_pre  = lm(n_nodes_pre~which(pre_ix))
mod_comp_post = lm(n_nodes_post~which(post_ix))

mod_comp_pre_exp  = lm(log(n_nodes_pre)~which(pre_ix))
mod_comp_post_exp = lm(log(n_nodes_post)~which(post_ix))

plot(d_seq[-1], n_nodes,
     xlab="Date", ylab="Size of giant component (# nodes)",
     main="Effect of the 2015 Blackout on the reddit graph"
     )
#lines(d_seq[-1][pre_ix], predict(mod_comp_pre), lty=2, col='blue')
#lines(d_seq[-1][post_ix][-26], predict(mod_comp_post), lty=2, col='red')
lines(d_seq[-1][pre_ix], exp(predict(mod_comp_pre_exp)), lty=2, col='blue')
lines(d_seq[-1][post_ix][!is.na(d_seq[-1][post_ix])], exp(predict(mod_comp_post_exp)), lty=2, col='red')

# By all accounts, the community is shrinking
coef(mod_comp_post)/coef(mod_comp_pre) # -3.422815 
# The community (qua graph) has shrunk about 3% since the blackout
# It was growing (qua graph) before the black, but it has flipped to shrinking (sign change)
# and is shrinking over 5x faster than it was previously growing.

# Before the blackout, the main component was growing at a rate of about 7.7 subreddits per week (+0.24%).
# Now, we're shrinking at a rate of about 26 subreddits per week (-0.85%). 

# As we observed analyzing the data from the perspective of active users, the community is shrinking about
# 3x faster than it was previously growing.

###############################################
# Investigate community growth wrt node count #
###############################################

gorder_no_isolates = function(g){
  d = degree(g)
  length(d[d>0])
}

n_nodes_no_isolates = sapply(graphs, gorder_no_isolates)

plot(d_seq[-1], n_nodes_no_isolates)

plot(d_seq[-1], sapply(graphs, ecount))

#####################################################################
# Compare my user activity rates against the public comment dataset #
#####################################################################

bq_user_activity = data.table(read.csv('time_series_anomaly_detection/fhoffa-bigquery_2015_active_users_month.csv'))
bq_user_activity[,month := ymd(paste0(month, "-28"))]
plot(unique(bq_user_activity), type='l',
     main="Monthly reddit active users from public comments",
     xlab="Date", ylab="Unique commenting users"
     )
# This plot tells a very different story. Not sure how to reconcile the general, unimpeded growth we're observing
# here with the reduction in graph-relevant comments

# Th best explanation I can come up with: there general userbase is still growing, but there is a certain class of users,
# i.e. users who are more likely to make comments containing links to external subreddits, that is participating less
# since the blackout. The other possible interpretation is that something is discouraging people from making these kinds
# of comments, but I'm not sure what the mechanism there could be.

mod_bq1 = lm(active_users~month, bq_user_activity[month<ymd("2015-06-29")])
mod_bq2 = lm(active_users~month, bq_user_activity[month>ymd("2015-06-29")])
#mod_bq3 = lm(log(active_users)~month, bq_user_activity) # the rate term is so small this is essentially a linear model
mod_bq3 = lm(active_users~month, bq_user_activity)
summary(mod_bq3) # .902 adj-r2
lines(bq_user_activity[month<ymd("2015-06-29"), month], predict(mod_bq1), lty=2, col='blue')
lines(bq_user_activity[month>ymd("2015-06-29"), month], predict(mod_bq2), lty=2, col='red')
lines(bq_user_activity$month, predict(mod_bq3), lty=2, col='purple')
#lines(bq_user_activity$month, exp(predict(mod_bq3)), lty=2, col='purple')

#mentions_monthly_activity = mentions[, list(active_users = length(unique(author))), by=list(year(date), month(date))]
setkey(mentions, date)
mentions_monthly_activity = mentions[, list(drange = max(date)-min(date)), by=list(author, year(date), month(date))] # this is silly slow
mentions_monthly_activity = mentions_monthly_activity[drange>days(7), list(active_users=length(unique(author))), by=list(year,month)]
mentions_monthly_activity[,date := ymd(paste(year,month, "28", "-"))]
mentions_monthly_activity[!(month%in%c(2,12,1))][order(date)][,#[date>=ymd("2014-06-01")][,
  plot(date, active_users, 
       type='l',
       xlab="Date", ylab="Unique users making 'subreddit mention' comments",
       main="Relative user activity constraining attention to 'subreddit mention' comments"
       )] 
abline(v=ymd("2015-07-04"), lty=2)
# constraining attention to these months looks more like what I was expecting to see, but the fact is: 

mod_users = lm(active_users~date, mentions_monthly_activity[date>=ymd("2012-01-01") & date<=ymd("2015-06-01")])
summary(mod_users)
#plot(mod_users)
lines(mentions_monthly_activity[date>=ymd("2012-01-01") & date<=ymd("2015-06-01"), date], predict(mod_users), lty=2, col='blue')


save(mentions_monthly_activity, file="rdata/mentions_monthly_activity.rdata")

###########################

# Measure community size as # of nodes with degree > 0 at each time step