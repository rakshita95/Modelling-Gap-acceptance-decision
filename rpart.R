library(plyr)
library(dplyr)
library(rpart)

classes = c("character","character","numeric","character","numeric","numeric","character","character")
transport = read.csv('E:/academics/sem 7/SVM-transport/data/data_trans.csv')

# Feature engineering
transport$Through.Veh. = revalue(transport$Through.Veh.,
                                 c("2w" = "2W",
                                   "AR" = "AUTO",
                                   "JEEP" = "CAR",
                                   "VAN" = "CAR",
                                   "BUS" = "TRUCK",
                                   "TEMPO" = "TRUCK",
                                   "MINI TRUCK (ACE)" = "TRUCK"))
transport$Veh = paste(transport$Sub.Vehicle, transport$Through.Veh., sep="_")

# Convert each predictor be either numeric or nominal (factor rather than character or logical)
chr.cols = transport %>% summarise_each(funs(is.character(.))) %>% unlist() %>% which() %>% names()
lgl.cols = transport %>% summarise_each(funs(is.logical(.))) %>% unlist() %>% which() %>% names()
transport = transport %>% mutate_each( funs(as.factor), one_of(chr.cols,lgl.cols))

#Test-Train split
train = transport[0:987,]
test = tail(transport,247)
# grow tree 
control_params <- rpart.control(cp= 0.0114,maxdepth=20)
fit <- rpart(Accpt.Reg ~ Through.Veh.+Speed..km.hr.+Spatial.Gap,
             method="class", data=train,control= control_params) #,minsplit = 2, minbucket = 1,cp=-1



#Plot tree
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(fit)
fit$variable.importance
printcp(fit)
bestcp <- fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
fit.pruned <- prune(fit, cp = bestcp)
fancyRpartPlot(fit.pruned)
min_xerror <-fit$cptable[which.min(fit$cptable[,"xerror"]),"xerror"] 

#prediction
preds <- predict(fit, test, type = c("class"))
confusion_matrix<-table(preds, test$Accpt.Reg)
rownames(confusion_matrix) <- paste("Actual", rownames(confusion_matrix), sep = ":")
colnames(confusion_matrix) <- paste("Pred", colnames(confusion_matrix), sep = ":")
print(confusion_matrix)
accuracy <-  sum(confusion_matrix[2,2],confusion_matrix[1,1])/sum(confusion_matrix)
print(accuracy)

#ROC
# library(ROCR)
# pred <- prediction(predict(fit.pruned, type = "prob")[, 2], train$Accpt.Reg)
# plot(performance(pred, "tpr", "fpr"))
# abline(0, 1, lty = 2)
# 
# x <- unlist(slot(a,"x.values")) 
# y <- unlist(slot(a,"y.values")) 

library(caret)
tc <- trainControl("cv",10, savePredictions = TRUE)
rpart.grid <- expand.grid(.cp=0.01)
train.rpart <- train(Accpt.Reg ~ Veh+Speed..km.hr.+Spatial.Gap, data=transport, method="rpart",trControl=tc,tuneGrid=rpart.grid)

print(train.rpart$results[1,2])
Confusion_matrix <- confusionMatrix(train.rpart)
print(Cofusion_matrix )
recall <- Confusion_matrix$table[1,1]/(Confusion_matrix$table[1,1]+Confusion_matrix$table[2,1])    # since it is a table, not data.frame
print(recall)
precision <-Confusion_matrix$table[1,1]/(Confusion_matrix$table[1,1]+Confusion_matrix$table[1,2])  
print(precision)
