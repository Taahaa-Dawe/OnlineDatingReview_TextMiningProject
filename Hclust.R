data <- read.csv("D:\\Boulder\\Text Mining\\DataProceesed.csv")

head(data)
data1 <- (data[,1:ncol(data)-1])

library(dplyr)

labels <- 0

labels <- paste(data$cluster, 1:length(d[,1]), sep = "_")

labels

rownames(data1) <- labels

dist_mat <- dist(data1, method = 'euclidean')

length(dist_mat)

hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg)


(My_m <- (as.matrix((t(data1)))))

(My_cosine_dist = 1-crossprod(My_m) /(sqrt(colSums(My_m^2)%*%t(colSums(My_m^2)))))



My_cosine_dist <- as.dist(My_cosine_dist)
HClust_Ward_CosSim_SmallCorp2 <- hclust(My_cosine_dist, method = "ward.D")

plot(HClust_Ward_CosSim_SmallCorp2, cex = 0.7, hang = -30, main = "Cosine Sim")

rect.hclust(HClust_Ward_CosSim_SmallCorp2, k = 2)