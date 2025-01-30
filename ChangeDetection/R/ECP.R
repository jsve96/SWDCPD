library(ecp)
library(hash)

current_dir <- getwd()

datasetpath <- paste(dirname(getwd()),"/datasets/has2023DataR/HASC_TS_",sep="")
print(datasetpath)
IDs = list(10,14,7,182,225,19,185,33,36,87,88,210,11,20,23,243,247,91,95,96,100,141,91,95,245)
lol <- list()
for (id in IDs){
  print(id)
  
  string = paste(paste(datasetpath,id,sep = ""),'.csv',sep="")
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


# Occupancy

datapath <-  paste(dirname(getwd()),"/datasets/Occupancy/Occupancy.csv",sep="")


X <- read.csv(datapath)
datapts <- X[,-1]

print('Start')
start_time <- Sys.time()
res <- e.divisive(datapts,R=30,sig.lvl=0.05,min.size = 400)
end_time <- Sys.time()
print(res$estimates)


