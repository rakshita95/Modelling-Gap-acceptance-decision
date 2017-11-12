
classes = c("character","character","numeric","character","numeric","numeric","character","numeric")

transport = read.csv('E:/academics/sem 7/SVM-transport/Data/DayandNightNew.csv')

#Change through veh names as desired
library(plyr)
library(dplyr)
#change these to what sir suggested...
transport$Through.Veh. = revalue(transport$Through.Veh.,
                                    c("2w" = "2W",
                                      "JEEP" = "CAR",
                                      "BUS" = "TRUCK",
                                      "AR" = "AUTO",
                                      "VAN" = "CAR",
                                      "MINI TRUCK (ACE)"="TRUCK",
                                      "TEMPO"="TRUCK"
                                      ))


## Let each predictor be either numeric or nominal (factor rather than character or logical)                        
chr.cols = transport %>% summarise_each(funs(is.character(.))) %>% unlist() %>% which() %>% names()
lgl.cols = transport %>% summarise_each(funs(is.logical(.))) %>% unlist() %>% which() %>% names()
transport = transport %>% mutate_each( funs(as.factor), one_of(chr.cols,lgl.cols))

# To remove column: trainData <- subset(trainData, select = -c(waterpoint_type_group))

# Classification Tree with rpart
library(rpart)

# grow tree 
fit <- rpart(Accpt.Reg ~ Through.Veh.+Speed..km.hr.+Lag.Gap+Spatial.Gap,
             method="class", data=transport,minsplit = 2, minbucket = 1,cp=-1)
#minspit: the minimum number of observations that must exist in a node in order for a split to be attempted
#minbucket: 
#cp: complexity parameter
printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, 
     main="Classification Tree for Transport")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# create postscript plot of tree 
post(fit, file = "c:/tree.ps", title = "Classification Tree for Kyphosis")
