library(arulesViz)
library(arules)

data <- read.transactions("D:/Boulder/Text Mining/Project/TransactionDataTextMining.csv",
                          rm.duplicates = FALSE,
                          format = "basket",  ##if you use "single" also use cols=c(1,2)
                          sep=",",  ## csv file
                          cols=1, skip = 1)
data
inspect(data[1:10])


FirstRule = arules::apriori(data, parameter = list(support=0.03, confidence = 0.8,
                                                    minlen=2),
                             appearance = list(default="lhs", rhs="dating"))

SortedRules1 <- sort(FirstRule, by="support", decreasing=TRUE)
inspect(SortedRules1)


(summary(SortedRules1))
plot(SortedRules1, method="graph", engine="htmlwidget")




SortedRules2 <- sort(FirstRule, by="confidence", decreasing=TRUE)
inspect(SortedRules2)


SortedRules3 <- sort(FirstRule, by="lift", decreasing=TRUE)
inspect(SortedRules3)