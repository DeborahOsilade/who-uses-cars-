
#libraries
library(DataExplorer)
library(ggplot2)
library(VIM)
library(caret)
library(car) 
library(MASS) 
library(caTools)
library(ROCR)
library(ineq)
library(e1071)
library(class)
library(gbm)          # basic implementation using AdaBoost
library(xgboost)      # a faster implementation of a gbm
library(ipred)
library(rpart) 
library(DMwR)


#variable changes 

table(Cars_dataset$Transport)
colnames(Cars_dataset)
colnames(Cars_dataset)<- make.names(colnames(Cars_dataset))
str(Cars_dataset)
Cars_dataset$Transport <- as.factor(Cars_dataset$Transport) #target variable has to be categorical
Cars_dataset$Gender <-  as.factor(Cars_dataset$Gender)  #gender is categorical 
levels(Cars_dataset$Transport)


#summary 

plot_intro(Cars_dataset) #22% discrete colums, shows missing variables 
plot_missing(Cars_dataset) # missing values in MBA 
#treating missing values 
Cars_dataset <- na.omit(Cars_dataset)
sum(Cars_dataset$Transport== '2 Wheeler')
sum(Cars_dataset$Gender== 'male')
summary(Cars_dataset)
levels(Cars_dataset$Age)


#univirate 

attach(Cars_dataset)
histogram(Salary,main='histogram of Salary') #shows outliers and skewed data 
histogram(Age, main= 'histogram of Age') # almost normally distributed 
barchart(Gender, main= 'Barplot of Gender') # more males than females in the company 
barchart(Transport,main = 'Barplot of Transport') # more people use public transport than cars 
histogram(Distance) # shows a normal distribution 
histogram(Work.Exp) # shows extreme years of work exp.
boxplot(Age, horizontal = TRUE, col = 'blue', main= 'boxplot of Age') # presence of outliers
boxplot(Salary,horizontal = TRUE, col = 'blue', main= 'boxplot of Salary') # shows low earners with extreme payments for higher earner in the company 
boxplot(Distance,horizontal = TRUE, col = 'blue', main= 'boxplot of Distance') #shows a few outliers 
boxplot(Work.Exp,horizontal = TRUE, col = 'blue', main= 'boxplot of Work.Exp')


#bivirate

plot_correlation(Cars_dataset,maxcat = 3L) #strong relationship between transport car and workexp and salary,distance,age,license. #age and work exp and salary.
plot(Salary,Work.Exp)
plot(Age,Salary)
boxplot(Age~Transport,horizontal = TRUE, col = 'blue', main= 'relationship between age and transport') #shows older people uses car 
boxplot(Work.Exp~Transport,horizontal = TRUE, col = 'blue', main= 'relationship between transport and work.exp')# staff with higher work experiences uses cars
boxplot(Salary~Transport,horizontal = TRUE, col = 'blue', main= 'relationship between transport and Salary')# higer salary earners use cars 
boxplot(Distance~Transport,horizontal = TRUE, col = 'blue', main= 'relationship between transport and distance')#people who use cars live far.



#treating outlier

attach(Cars_dataset)
quantile(Salary,c(0.01,0.02,0.03,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
plot(density(Salary),main="Salary") #shows skewness to the right
qqnorm(Salary) 
#capping at 99% percentile
Salary[which(Salary<41.92)] <- 41.92



quantile(Work.Exp,c(0.01,0.02,0.03,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
Work.Exp[which(Work.Exp<21)] <- 21
plot(density(Work.Exp),main="Work.Exp")
qqnorm(Work.Exp)


quantile(Age,c(0.01,0.02,0.03,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
plot(density(Age),main="Age")
qqnorm(Age) 



#binning variable

levels(Cars_dataset$Transport)
Cars_dataset$Transport <- factor(Cars_dataset$Transport,c("2Wheeler","Public Transport","Car")) #reordering the variables 
levels(Cars_dataset$Transport)[1:2] <- "Public transport"  #binning 2 wheeler and public transport as public transport 
attach(Cars_dataset)
levels(Gender)
levels(Cars_dataset$Gender) <- c("0", "1")
levels(Cars_dataset$Transport) <- c("0", "1")
str(Cars_dataset)


#building a logistic model with the real dataset 

log.reg <- glm(Transport~., data=Cars_dataset, family=binomial(link="logit"))
summary(log.reg)
#the model indicates distance as the most important variable
# Check variable importance 
varImp(log.reg)
model_2 <- glm(Transport~Distance, data=Cars_dataset, family=binomial(link="logit"))
summary(model_2)
model_3 <- glm(Transport~Distance+Salary+license, data=Cars_dataset, family=binomial(link="logit"))
summary(model_3)
plot (model_3)
vif(model_3)



#divide data into train and test 

# Divide data in "70:30"
set.seed(100)
Part_car_data <- sample(1:nrow(Cars_dataset), 0.7*nrow(Cars_dataset))
# Training set
train_Cardata <- Cars_dataset[Part_car_data,]
str(train_Cardata)
# Test set
test_Car_data <- Cars_dataset[-Part_car_data,]
dim(train_Cardata)
dim(test_Car_data)
table(train_Cardata$Transport) # 0 - 268 , 1 - 23 on the train set 
table(test_Car_data$Transport) # 0 - 114 , 1 - 12 on the test set 



#running the logistic model on the train data

model_31 <- glm(Transport~Distance+Salary+license, data=train_Cardata, family=binomial)
summary(model_31)
vif(model_31) #vif is low 
model_31 <- train(Transport~Distance+Salary+license,data=train_Cardata,method="glm", family="binomial")
table(test_Car_data$Transport,test_Car_data$log.pred>0.5)



model20 <- train(Transport~.,data=train_Cardata,method="glm", family="binomial")
summary(model20)



model20_r <- predict(model20,newdata = train_Cardata,type = "raw")
caret::confusionMatrix(model20_r,train_Cardata$Transport)



model31_p <- predict(model20,newdata = test_Car_data,type = "raw")
caret::confusionMatrix(model31_p,test_Car_data$Transport)


#checking model performance using ROC Curve

lr_predictions_prob <- predict(model_31, newdata = test_Car_data, type = "response")
# Creating the prediction object using ROCR library
lr_pred_obj = prediction(lr_predictions_prob, test_Car_data$Transport)
# Plotting the ROC curve 
roc_LR = performance(lr_pred_obj, "tpr", "fpr")
plot(roc_LR, colorize=TRUE)
# Plotting the PR curve
precision_recall_LR<- performance(lr_pred_obj, "ppv", "tpr")
plot(precision_recall_LR, xlab = "Recall", ylab = "Precision",colorize=TRUE)
# Computing the area under the curve
auc = performance(lr_pred_obj,"auc"); 
auc = as.numeric(auc@y.values)
auc



# naive bayes

#for the model
NB.Model <- naiveBayes(Transport~., data=train_Cardata,trControl = fitControl)
summary(NB.Model)


#model performance on naive bayes

NB_Test_Predictions <- predict(NB.Model,newdata=test_Car_data)
confusionMatrix(NB_Test_Predictions , test_Car_data$Transport,positive = "1")
NB_Train_Predictions <- predict(NB.Model,newdata=train_Cardata)
confusionMatrix(NB_Train_Predictions , train_Cardata$Transport,positive = "1")


###cart model

library(rpart)
Car_ctrl_parameter = rpart.control( minbucket = 10, cp = 0, xval = 5)
Car_Cart_Model<- rpart(formula = Transport~., data = Cars_dataset[,1:9], 
                       method = "class",control = Car_ctrl_parameter)
library(rattle)
fancyRpartPlot(Car_Cart_Model, cex= 0.6)



Car_cart_predictions_test <- predict(Car_Cart_Model , newdata = test_Car_data, type = "class")
Car_cart_predictions_train <- predict(Car_Cart_Model , newdata = train_Cardata, type = "class")



#model performance for CART

#cONFUSION MATRIX 
library(caret)
Cart_trainmatrix <- confusionMatrix(Car_cart_predictions_train,train_Cardata$Transport)
Cart_matrix
Cart_testmatrix <- confusionMatrix(Car_cart_predictions_test,test_Car_data$Transport)
Cart_testmatrix



#Randomforests model

set.seed(100)
library(randomForest)
Car_rndForest <- randomForest(Transport~ ., data = train_Cardata,ntree=101,mtry=3, nodesize=5,importance=TRUE) 
##Print the model to see the OOB and error rate
print(Car_rndForest)
importance(Car_rndForest)
varImpPlot(Car_rndForest,type=2)
plot(importance(Car_rndForest))
plot(Car_rndForest)  


#model performance of random forest model

Car_rnd_predictions_test_class <- predict(Car_rndForest , newdata = test_Car_data, type = "response")


#confusion matrix 

rf_matrix <- confusionMatrix(Car_rnd_predictions_test_class, test_Car_data$Transport,positive = "1")
rf_matrix



#knn model

#scale the data

scaleValues <- preProcess(train_Cardata, method = c("center", "scale"))
trainTransformed <- predict(scaleValues, train_Cardata)
testTransformed <- predict(scaleValues, test_Car_data)


#Knn model

knn_model <- train(Transport ~ ., data = trainTransformed,
                   method = "knn",
                   tuneLength = 20) 
knn_model
plot(knn_model)



#model performance knn

knn_predictions_train <- predict(knn_model, newdata = trainTransformed, type = "raw")
knn_predictions_test <- predict(knn_model, newdata = testTransformed, type = "raw")



confusionMatrix(knn_predictions_train, train_Cardata$Transport)
confusionMatrix(knn_predictions_test, test_Car_data$Transport)



#bagging

Cars_Bagging<- bagging(Transport~.,
                       data=train_Cardata,
                       control=rpart.control(maxdepth=5, minsplit=4))
varImp(Cars_Bagging)
plot(varImp(Cars_Bagging))
test_Car_data$pred_class <- predict(Cars_Bagging, test_Car_data)
View(test_Car_data)
confusionMatrix(data=factor(test_Car_data$pred_class  ),
                reference=factor(test_Car_data$Transport)  ,
                positive='1')



#gradient boosting 

gbm_model <- train(Transport ~ ., data = train_Cardata,
                   method = "gbm",
                   
                   verbose = FALSE)
plot(gbm_model)
plot(varImp(gbm_model))
gbm_predictions_test <- predict(gbm_model, newdata = test_Car_data, type = "raw")
confusionMatrix(gbm_predictions_test, test_Car_data$Transport)


#xgboost

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)
xgb.grid <- expand.grid(nrounds = 100,
                        eta = c(0.01),
                        max_depth = c(2,4),
                        gamma = 0,               
                        colsample_bytree = 1,    
                        min_child_weight = 1,    
                        subsample = 1            
)
xgb_model <-train(Transport~.,
                  data=train_Cardata,
                  method="xgbTree",
                  trControl=cv.ctrl,
                  tuneGrid=xgb.grid,
                  verbose=T,
                  nthread = 2
)


#model comparison

# Compare model performances using resample()
models_to_compare <- resamples(list(Logistic_Regression = model_31 , 
                                    Navie_Bayes = NB.Model, 
                                    KNN = knn_model, 
                                    CART_Decision_tree = Car_Cart_Model, 
                                    Random_Forest =Car_rndForest ,
                                    Gradient_boosting = gbm_model,
                                    
                                    
))
# Summary of the models performances
summary(models_to_compare)