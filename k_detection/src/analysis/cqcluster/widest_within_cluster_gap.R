if(!require('fpc')) {
    install.packages('fpc')
    library('fpc')
}

# NOT RUN {
require(graphics)



###########################################################################################

path_to_dir = getwd()
path_to_file = paste(path_to_dir , "/analysis/cqcluster/k_means_input.csv", sep = "")
data <- read.csv(path_to_file)
print(nrow(data))
print(ncol(data))
print(data)
print(class(data))


if (FALSE){
    set.seed(20000)
    options(digits=3)
    
    x <- rbind(matrix(rnorm(10, sd = 0.3), ncol = 2),
                matrix(rnorm(100, mean = 1, sd = 0.3), ncol = 2))
    colnames(x) <- c("x", "y")
    class(x)
    print(x)
    x <- dist(x)
    class(x)

    complete3 <- cutree(hclust(x),3)
    print(complete3)

    # cqcluster.stats(dface,complete3,
    #          alt.clustering=as.integer(attr(face,"grouping")))
  



    x <- rbind(matrix(rnorm(10, sd = 0.3), ncol = 2),
            matrix(rnorm(100, mean = 1, sd = 0.3), ncol = 2))
    colnames(x) <- c("x", "y")
    print(x)
    (cl <- kmeans(x, 2))


    standardisation="max"
    if (is.numeric(standardisation)) stan <- standardisation
    else
        stan <- switch(standardisation,
                    max=max(d),
                    ave=mean(d),
                    q90=quantile(d,0.9),
                    1)


    clustering ? 
    cn <- max(clustering)
    cwn <- cn
    cwidegap <- rep(0, cwn)
    for (i in 1:cwn) if (sum(clustering == i) > 1) 
        cwidegap[i] <- max(hclust(as.dist(dmat[clustering == i, clustering == i]), method = "single")$height)
    if (averagegap)
        widestgap <- mean(cwidegap)/stan
    else    
        widestgap <- max(cwidegap)/stan
}