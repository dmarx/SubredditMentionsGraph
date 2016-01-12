install.packages("igraphdata")
library(lubridate)
library(igraph)
library(igraphdata)
data(enron)

E(enron)$truncdate = floor_date(ymd_hms(E(enron)$Time), unit="day")

dates = unique(E(enron)$truncdate)
bad_date = 315522000
dmin = min(dates[dates>bad_date])
dmax = max(dates)
plot(table(E(enron)$truncdate), xlim=c(dmin, dmax)) 
# Do we even need to represent this as a graph to spot the outlier dates?
dates2 = sort(unique(dates[dates>bad_date]))

start = Sys.time()
graphs = list()
i=0
for(d in dates2){
  i=i+1
  if(i%%100==0)print(i)
  g = delete.edges(enron, which(E(enron)$truncdate!=d))
  graphs[as.character(d)] = list(g)
}
end = Sys.time()
elaps = end - start
print(elaps) # 20 seconds

start = Sys.time()
test = scan_stat(graphs, tau=30)
end = Sys.time()
elaps = end - start
print(elaps) # 10 seconds


d_test = as.POSIXct(dates2, origin='1970-1-1')
plot(d_test, test$stat, type='l')
rug(ymd("2001-03-20"), col="red", lty=2)
rug(ymd("2001-8-22"), col='purple', lty=2)
rug(ymd("2001-10-14"), col='blue', lty=2)

get_peak_details = function(peak_ix){
  peak_date = d_test[-1][peak_ix]
  node_id = test$arg_max_v[peak_ix]
  node = V(graphs[[peak_ix]])[node_id]
  list(date=peak_date, node=node$Name, node_id=node_id)
}

details = data.table(t(sapply(which(!is.na(test$stat)), get_peak_details)))
details$stat = test$stat[!is.na(test$stat)]
details[order(stat, decreasing=TRUE)][1:3]

#                   date           node node_id     stat
# 1: 2001-03-22 19:00:00  Jim Schwieger      71 4692.000
# 2: 2001-08-23 20:00:00    Kenneth Lay      95 3856.000
# 3: 2001-03-20 19:00:00  Thomas Martin     174 2969.867

