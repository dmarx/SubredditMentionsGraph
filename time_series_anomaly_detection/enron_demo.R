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
print(elaps) # 6.5 min

start = Sys.time()
test = scan_stat(graphs, tau=30)
end = Sys.time()
elaps = end - start
print(elaps) # 3min


d_test = as.POSIXct(dates2, origin='1970-1-1')
plot(d_test, test$stat, type='l')
rug(ymd("2001-03-20"), col="red", lty=2)
rug(ymd("2001-8-22"), col='purple', lty=2)
rug(ymd("2001-10-14"), col='blue', lty=2)

