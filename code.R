library(dplyr)
library(tidyverse)
library(caret)
library(MASS)
library(ggplot2)
library(class)
library(gmodels)
library(PerformanceAnalytics)
library(boot)
library(Metrics)
library(plyr)
library(klaR)
library(DAAG)


#install.packages('Metrics')

data <- data.frame(read.delim("C://Users//vnino//Desktop//U//MA321 Applied Statistics//Group project//NEW DATA wmsa.txt")) 

#(1) Exploratory data analysis ########################################################
########################################################################################

#SUMMARY
summary(data)
## All variables are numerical. Lag1 to Lag 4 have the same range and scale.  
## Volume takes only positive values. See if it would be useful to re-scale (mean=0). 
## Distribution of classes: 544 Up (55.2%) /441 Down (44.8%)

#MISSING VALUES
sapply(data, function(x) sum(is.na(x)))
## There are no missing values in the data.

#SPOT OUTLIERS
#observations that lie outside 1.5 * IQR, where IQR, 
#the 'Inter Quartile Range' is the difference between 75th and 25th quartiles. 
#remove(outlier_values)

outlier_values_lag1 <- boxplot.stats(data$Lag1)$out  
outlier_values_lag2 <- boxplot.stats(data$Lag2)$out  
outlier_values_lag3 <- boxplot.stats(data$Lag3)$out  
outlier_values_lag4 <- boxplot.stats(data$Lag4)$out  
boxplot(Lag1~Direction,data=data, main="Lag1", boxwex=0.1)
boxplot(Lag2~Direction,data=data, main="Lag2", boxwex=0.1)
boxplot(Lag3~Direction,data=data, main="Lag3", boxwex=0.1)
boxplot(Lag4~Direction,data=data, main="Lag4", boxwex=0.1)

#CORRELATION
cor(data[,c(-1,-9)])
chart.Correlation(data[,c(-1,-9)], histogram=TRUE, pch=19)
#There is no strong correlation of the explanatory variables with the response variable (This Week).
#The scatter plots show no clear relationship between these variable either. 
#There is a strong correlation between year and volume, indicating that volume increases over time.
# This can be observed in the scatter plot, where one can see an increasing of the volume over time,
# especially in the last years.
# We should not expect our models to perform exceedingly well because the 
#is little relationship between the predictors and response.

#(2) Logistic Regression 5 variables ###################################################
########################################################################################

# VALIDATION APPROACH (TRAINING+VALIDATION) #####################################################

#We define the splitting
set.seed(1234)
training.samples <- data$Direction %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data <- data[training.samples, ]
test.data <- data[-training.samples, ]

summary(train.data) #CLASS DIST:UP 55.3%/ DOWN 44.7%
summary(test.data) #CLASS DIST:UP 55.1%/ DOWN 44.9%

#We fit using the training data:
glm.fittrain <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Volume , data = train.data, family = binomial)
summary(glm.fittrain)

#We compute the performance scores on the training data:
glm.probstrain <- predict(glm.fittrain,newdata=train.data, type = "response")
glm.predtrain <- ifelse(glm.probstrain > 0.5, "Up", "Down") 
table(glm.predtrain,train.data$Direction) # confusion matrix
# Accuracy: 55.64%.
# SENSITIVITY	82.3% SPECIFICITY: 22.7%

#We compute the performance scores on the validation data (test):
glm.probstest <- predict(glm.fittrain,newdata=test.data, type = "response")
glm.classtest <- ifelse(glm.probstest > 0.5, "Up", "Down") 
table(glm.classtest,test.data$Direction) # confusion matrix
# Accuracy: 53.57%. 
# SENSITIVITY	81.5% SPECIFICITY 19.3%
# we add the numeric Direction field to the test dataset to use it when calculating the MSE.
test.data$NDirection <- as.numeric(ifelse(test.data$Direction == "Up", 1, 0))
mse(glm.probstest,test.data$NDirection) # Mean Squared Error: 0.255

# CROSS-VALIDATION LEAVE-ONE-OUT ######################################################
glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Volume , data = data, family = binomial)
cv.err=cv.glm(data, glm.fit)
cv.err$delta # Average Mean Squared Error: 0.248

#(3) LDA 5 variables ###################################################################
########################################################################################

# ALL DATA #############################################################################

#We use the same splitting defined in previous point

#We fit using the training data:
lda.fittrain <-  lda( Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Volume, data=train.data)
summary(lda.fittrain)

#We compute the performance scores on the training data:
lda.predtrain<-predict(lda.fittrain,train.data)
lda.probstrain<- lda.predtrain$posterior[,2]
lda.classtrain <- lda.predtrain$class
table(lda.classtrain,train.data$Direction) 
# Accuracy: 55.5%. 
# SENSITIVITY	82.6%. SPECIFICITY 22.1%

#We compute the performance scores on the validation data (test):
lda.predtest<-predict(lda.fittrain,test.data)
lda.probstest<- lda.predtest$posterior[,2]
lda.classtest <- lda.predtest$class
table(lda.classtest,test.data$Direction)
# Accuracy: 53.6%. 
# SENSITIVITY	81.5%. SPECIFICITY 19.3%
mse(lda.probstest,test.data$NDirection) # Mean Squared Error: 0.255

# Stacked Histogram Plot of the LDA Values
plot(lda.fittrain, dimen = 1, type = "b")

# CROSS-VALIDATION LEAVE-ONE-OUT ######################################################

# We fit a model with all the data activating CV=TRUE to activate the Leave-one-out CV.
lda.fit.cv <- lda( Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Volume, data=data, CV=TRUE) 
# We add the numeric Direction field to the original dataset to use it in the calculation of MSE.
data$NDirection <- as.numeric(ifelse(data$Direction == "Up", 1, 0)) 
# We compute the MSE
mse(lda.fit.cv$posterior[,2], data$NDirection) # #1=Down, 2=Up, Mean Squared Error: 0.247

# (4) KNN ##############################################################################
########################################################################################

# VALIDATION APPROACH (TRAINING+VALIDATION) ############################################

#Normalization Function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

# Normalize features and divide the datasets in features and labels.
norm.train.data <- as.data.frame(lapply(train.data[3:7], normalize)) 
norm.test.data <- as.data.frame(lapply(test.data[3:7], normalize)) 
labels_train <- data[training.samples, ]
labels_train <- labels_train[, (names(labels_train) %in% c("Direction"))]
labels_test <- data[-training.samples, ]
labels_test <- labels_test[, (names(labels_test) %in% c("Direction"))]

#Defining the optimum K
ACC=0
for (i in seq(1, by = 2, len = 20)){ # We use only odd numbers to avoid ties.
set.seed(1234)
knn.mod <- knn(train=norm.train.data, test=norm.test.data, cl=labels_train, k=i, prob=TRUE)
ACC[i]=sum(labels_test == knn.mod)/NROW(labels_test)
cat(i,'=', ACC[i],'\n')}

plot(ACC[!is.na(ACC)], type="b", xlab="K- Value",ylab="Accuracy level")  # to plot % accuracy wrt to k-value
max(ACC[!is.na(ACC)])  # max achieved accuracy
which.max(ACC)  # k value for the max accuracy

#knn classifier using the best K parameter
knn.mod.K19 <- knn(train = norm.train.data, test = norm.test.data, cl = labels_train, k=19)

#confusion matrix (chi-squared contribution is turned off)
confusionMatrix(knn.mod.K19,labels_test)


# CROSS-VALIDATION LEAVE-ONE-OUT ######################################################

#Load complete data for the leave one out validation and divide the dataset in features and labels.
data_features <- as.data.frame(lapply(data[3:7], normalize))
data_labels <- data$Direction 

#We only do leave-one-out validation with the optimal K
knn.mod.K19.cv<- knn.cv(data_features, data_labels, k = 19, prob=TRUE)
# MSE for LOOCV: 
data_labels_num <- as.numeric(ifelse(data_labels == "Up", 1, 0))
# We compute a dataframe that has as a first column the numeric labels and as a second 
# column the probability of the assigned class (obtained form attr prob)
labels_prob <- as.data.frame(cbind(labels=data_labels_num, prob=attr(knn.mod.K19.cv,'prob')))
# We obtain the estimate for Pr(Y=1) 
labels_prob$prob_Up <- as.numeric(ifelse(labels_prob$labels == 1,labels_prob$prob,1-labels_prob$prob))

mean((labels_prob$prob_Up-labels_prob$labels)^2) # Mean Squared Error: 0.160

mse(labels_prob$prob_Up,labels_prob$labels)

# (5) COMBINATION OF PREDICTORS ########################################################
########################################################################################
## TRATINING  WITH DIFFERENT COMBINATION OF PREDICTORS ## PROPOSED FEATURES:::
## SUM(LAGi) i=1,2
## LAG1/LAG2
## Log(volume)
## (Volume_i/Volume_i-1)-1

data2=data
data2$sum_lag_2=data2$Lag1+data2$Lag2
data2$prod_lag_2=data2$Lag1*data2$Lag2

data2$dif_lag2=data2$Lag1/data2$Lag2
data2$log_Volume=log(data2$Volume)

for (i in 2:length(data2$Volume)){data2$last_Volume[i]=data2$Volume[i-1]}
data2$last_Volume[1]=data2$Volume[1]
data2$dif_Volume=(data2$Volume/data2$last_Volume)-1

## Same partition as before
set.seed(1234)
training.samples <- data2$Direction %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data2 <- data2[training.samples, ]
test.data2 <- data2[-training.samples, ]

## Training Log regresion
glm.extended <- glm(Direction ~ Lag1 + Lag2 + Volume + dif_Volume, data = train.data2, family = binomial)
summary(glm.extended)

# We compute the performance scores on the validation data (test):
glm.probsext <- predict(glm.extended,newdata=test.data2, type = "response")
glm.predtextest <- ifelse(glm.probsext > 0.5, "Up", "Down") 
table(glm.predtextest,test.data2$Direction) # confusion matrix
mean(glm.predtextest == test.data2$Direction) # Accuracy: 53.6%.

### TRAINING LDA for Direction ~ Lag1 + Lag2 + Volume + dif_Volume
lda.fit = lda( Direction ~ Lag1 + Lag2 + Volume + dif_Volume, data=train.data2)
lda.pred = predict( lda.fit,test.data2, type = "response")
CM = table( predicted=lda.pred$class, truth=test.data2$Direction )
print( CM ) # Accuracy: 53.0%.
