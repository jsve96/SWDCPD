library(ecp)
library(hash)
library(jsonlite)

data <- fromJSON("C:/Users/Sven Jacob/Documents/Github/SWDCPD/ChangeDetection/data.json")

lol <- list()
id <- 0
for (subdata in data){
  datapts<-subdata$data
  print("CPS")
  print(subdata$labels)
  start_time <- Sys.time()
  res <- e.divisive(datapts,R=199,sig.lvl = 0.05)
  end_time <- Sys.time()
  print(res$estimates)
  print(end_time - start_time)
  id_c <- as.character(id)
  print(id_c)
  lol[[id_c]] <- res$estimates
  id <- id+1
   
  
}
  
write(toJSON(lol, pretty = TRUE), file = "MNIST_Edivise.json")


