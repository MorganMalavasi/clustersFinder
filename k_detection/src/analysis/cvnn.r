cvnn <- function(d=NULL,clusterings,k=5){
  d <- as.matrix(d)
  n <- nrow(d)
  neighb <- matrix(0,nrow=n,ncol=k)
  for (i in 1:n)
    neighb[i,] <- order(d[i,])[2:(k+1)]
  cnums <- sep <- comp <- sepf <- compf <- numeric(0)
  lc <- length(clusterings)
  for (i in 1:lc){
    cnums[i] <- max(clusterings[[i]])
    comp[i] <- nji <- 0
    sepj <- numeric(0)
    for (j in 1:cnums[i]){
      nj <- sum(clusterings[[i]]==j)
      compj <- sum(d[clusterings[[i]]==j,clusterings[[i]]==j])
      nji <- nji+nj*(nj-1)
      comp[i] <- comp[i]+compj
#      cat("i=",i," j=",j,"compj=",compj,"\n")
      sepj[j] <-  sum(clusterings[[i]][as.vector(neighb[(1:n)[clusterings[[i]]==j],])]!=j)/(k*nj)
    }
    sep[i] <- max(sepj)
    comp[i] <- comp[i]/nji
  }
  maxsep <- max(sep)
  maxcomp <- max(comp)
  sepf <- sep/maxsep
  compf <- comp/maxcomp
  cvnnindex <- sepf+compf
  out <- list(cvnnindex=cvnnindex,sep=sep,comp=comp)
}
    