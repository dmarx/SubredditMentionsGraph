bq_path = "E:/Projects/SubredditMentionsGraph/mentions_from_bigquery"

library(data.table)
mentions_bq <- lapply(list.files(bq_path, full.names=TRUE), fread)
mentions_bq = rbindlist(mentions_bq)
mentions_bq[,.N] - mentions_bq[author!='[deleted]',.N] # 591345
setnames(mentions_bq, c("source","target"), c("Source","Target"))

mentions = mentions_bq

