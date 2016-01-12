library(RSQLite)
library(data.table)
library(lubridate)
db_path = "E:/Projects/SubredditMentionsGraph/subreddit_mentions.db"
conn = dbConnect(SQLite(), db_path)
#dbListTables(conn)
#dbGetQuery(conn, "select * from mentions limit 10")


toEpoch = function(val){as.POSIXct(val, origin="1970-01-01")}

mentions = data.table(dbGetQuery(conn, "select * from mentions"))
setnames(mentions, c("source_subr_lwr", "target_subr_lwr"), c("Source", "Target"))
setkey(mentions, created_utc)

nodelist = mentions[,union(Source, Target)]

#drange = dbGetQuery(conn, "
#  select min(created_utc) d_min, max(created_utc) d_max from mentions
#  ")
drange = mentions[,list(d_min=min(created_utc), d_max=max(created_utc))]
drange = lapply(drange, toEpoch)
drange$d_min = floor_date(drange$d_min, unit="day") + days(1)
drange$d_max = floor_date(drange$d_max, unit="day")

d_seq = seq(from=drange$d_min, to=drange$d_max, by="week")
d_seq_numeric = as.numeric(d_seq)

graphs=list()
for(i in 2:length(d_seq)){
  d1 = as.numeric(d_seq[i-1])
  d2 = as.numeric(d_seq[i])
  print(c(i,d2))
  # Can't get bind variables to work. I'm not proud.
#   sql = paste("select * from mentions 
#               where created_utc > ", d1, 
#               " and created_utc <= ", d2)
#   records = data.table(dbGetQuery(conn, sql))
#   setnames(records, c("source_subr_lwr", "target_subr_lwr"), c("Source", "Target"))
  records = mentions[d1<created_utc & created_utc<=d2][,.(Source, Target, author)]
  #records = records[,.(Source, Target, author)]
  setkey(records, Source, Target)
  edges = records[,list(Weight=length(unique(author))), by=list(Source, Target)]
  g = graph.data.frame(edges, directed=TRUE, vertices=nodelist)
  graphs[as.character(d2)]=list(g)
}

start = Sys.time()
test = scan_stat(graphs, tau=8) # Each graph is 1 week, so this is approx 2 months
end = Sys.time()
elaps = end - start
print(elaps) # 7min


plot(d_seq[-1], test$stat, type='l') # oh baby!
abline(v=ymd("2015-06-10"), col="red", lty=2) # FPH banned
abline(v=ymd("2015-07-02"), col='purple', lty=2) # Victoria fired / 2015 Blackout
#abline(v=ymd("2015-07-10"), col='blue', lty=2)  # Ellen Pao steps down, Huffman appointed
abline(v=ymd("2015-11-20"), col='blue', lty=2)  # Privacy policy changes announced
#abline(v=ymd("2015-08-13"), col='blue', lty=2)  # Reddit bans /r/watchpeopledie in germany
abline(v=ymd("2015-08-05"), col='blue', lty=2)  # Reddit bans /r/coontown and other racist subs https://www.reddit.com/r/announcements/comments/3fx2au/content_policy_update/ctsqobs ## I'm not convinced this is it.


get_peak_details = function(peak_ix){
  peak_date = d_seq[-1][peak_ix]
  node_id = test$arg_max_v[peak_ix]
  node = V(graphs[[peak_ix]])[node_id] # user_history_bot ... wtf? Weirdness.
  list(date=peak_date, node=node$name)
}


# Identify the most chaotic dates (weeks) and neighborhoods
details = data.table(t(sapply(which(!is.na(test$stat)), get_peak_details)))
details$stat = test$stat[!is.na(test$stat)]
details[order(stat, decreasing=TRUE)]

# pretty sure this returns the size of the giant component
count_connected_nodes =  function(g){length(component_distribution(g))} 
n_nodes = sapply(graphs, count_connected_nodes)
plot(d_seq[-1], n_nodes)
# It looks like the size of the giant component has actually been *decreasing* over the past year. Very strange.

# Daily count of unique users making subreddit mentioning comments as a proxy
# for quantifying the size of the active userbase overtime (i.e. is the community shrinking?)
active_users = mentions[,list(count=length(unique(author))), by=list(year, month, day, wday)]
active_users[,date:=ymd(paste(year, month, day, sep='-'))]
active_users$index = 1:nrow(active_users)-1
active_users[,plot(date, count, type='l')] # oh shit... definitely shrinking, but plateauing. Looks like it's actually shrunk by about half
mod = lm(count~index, active_users)
mod2 = lm(count~date, active_users) # just for display purposes
abline(mod2, lty=2, col='blue')
summary(mod)
summary(mod)
coef(mod) 
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