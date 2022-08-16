if(!require('fpc')) {
    install.packages('fpc')
    library('fpc')
}
###########################################################################################

path_to_dir = getwd()
path_to_file_samples = paste(path_to_dir , "/analysis/cqcluster/k_means_input.csv", sep = "")
samples <- read.csv(path_to_file_samples)

path_to_file_labels = paste(path_to_dir , "/analysis/cqcluster/labels_input.csv", sep = "")
labels <- read.csv(path_to_file_labels)

path_to_file_labels2 = paste(path_to_dir , "/analysis/cqcluster/labels_input_2.csv", sep = "")
labels2 <- read.csv(path_to_file_labels2)

clustering1 <- labels$X0
clustering2 <- labels2$X0

value <- cvnn(dist(samples),list(clustering1, clustering2))
print(value)