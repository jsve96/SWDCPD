library(ecp)
library(hash)



IDs = list(10,14,7,182,225,19,185,33,36,87,88,210,11,20,23,243,247,91,95,96,100,141,91,95,245)
lol <- list()
for (id in IDs){
  print(id)
  
  string = paste(paste("C:/Users/Sven Jacob/Documents/Github/SWDCPD/Appendix/HASC_TS_",id,sep = ""),'.csv',sep="")
  X<-  read.csv(string)
  datapts <- X[,-1]
  print('Start')
  start_time <- Sys.time()
  res <- e.divisive(datapts,R=199,min.size = 500)
  end_time <- Sys.time()
  print(end_time - start_time)
  print(res$estimates)
  id_c <- as.character(id)
  print(id_c)
  lol[[id_c]] <- res$estimates
  print(lol)
  #hashmap$id_c <- res$estimates
}



library(jsonlite)

# Save the list of lists as JSON
write(toJSON(lol, pretty = TRUE), file = "HASC_EdiviseR199MINSIZE500.json")




X <- read.csv("C:/Users/Sven Jacob/Documents/Github/SWDCPD/ChangeDetection/Occupancy.csv")
datapts <- X[,-1]

print('Start')
start_time <- Sys.time()
res <- e.divisive(datapts,R=30,sig.lvl=0.05,min.size = 400)
end_time <- Sys.time()
print(res$estimates)



X <- read.csv("C:/Users/Sven Jacob/Documents/Github/SWDCPD/ChangeDetection/Syn1.csv")
datapts <- X[,-1]

print('Start')
start_time <- Sys.time()
res <- e.divisive(datapts,R=30,sig.lvl=0.05)
end_time <- Sys.time()
print(res$estimates)
