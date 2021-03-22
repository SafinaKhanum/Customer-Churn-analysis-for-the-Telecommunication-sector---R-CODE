The dataset used for this project is available on the below given link

https://www.kaggle.com/blastchar/telco-customer-churn



#Importing and Exploring the raw data set

#To clear all previous work from the R environment

rm(list=ls(all=T))

#Installing required packages and calling the respective libraries
library(DataExplorer)
library(ggplot2)
library(Hmisc)
library(car)
library(e1071)
library(reshape2)
library(rpart)
library(rpart.plot)
install.packages("digest")
library(modeest)
library(dplyr)
library(DMwR)
library(caret)
library(ROSE)
library(MLmetrics)
library(ROCR)
library(unbalanced)
library(FFTrees)
library(digest)
install.packages("C://Users//Admin//Downloads//FFTrees_1.5.5.tar.gz", repos = NULL, type="source")
install.packages("circlize")
library(circlize)
library(rattle)
library(RColorBrewer)
library(randomForest)
library(survival)
library(survminer)


#Importing the raw data
Telcom_data<-read.csv(choose.files(),header = TRUE)
View(Telcom_data)

#Data dimension
dim(Telcom_data)

#The raw data set has 7043 observations (7043 customers) and 21 variables


#Variable names in the data set
names(Telcom_data)

#Viewing the first and last 10 observations of the data set
head(Telcom_data,10)
tail(Telcom_data,10)

#Raw data set summary
introduce(Telcom_data)

#Data structure
str(Telcom_data)

#Number of unique values under each variable
apply(Telcom_data,2,function(x) length(unique(x)))

#Checking for duplicate records in the data set
sum(duplicated(Telcom_data$customerID))

#There are no duplicate observations in the data set

#Checking for missing values in the data set
plot_intro(Telcom_data) 

sum(is.na(Telcom_data))

#The data set has 11 missing values

#Checking column wise % of missing values
colSums(is.na(Telcom_data))/nrow(Telcom_data)*100

#The column total charges has 11 missing values

#No column has >30% of missing values hence all of them can be retained for further analysis


#Descriptive statistics for the continuous variables in the data

#Creating a data frame of only the continuous variables
num_df<-subset(Telcom_data,select = c(tenure,MonthlyCharges,TotalCharges))
View(num_df)

summary(num_df)

#Customers whose tenure is 0 months
zero_months<-Telcom_data[(Telcom_data$tenure == 0),]
dim(zero_months)
View(zero_months)

#Distribution of all numeric/continuous variables in the data set
plot_histogram(num_df)
plot_density(num_df)

#Skewness co-efficient
skewness(num_df$MonthlyCharges)
skewness(num_df$tenure)
skewness(missing_removed$TotalCharges) #Doesnt show with num_df due to presence of missing values. The processed data should be used instead.


#DATA PRE-PROCESSING

#Converting all categorical variables from character to factor
Telcom_data$gender<-as.factor(Telcom_data$gender)
Telcom_data$SeniorCitizen<-as.factor(Telcom_data$SeniorCitizen)
Telcom_data$Partner<-as.factor(Telcom_data$Partner)
Telcom_data$Dependents<-as.factor(Telcom_data$Dependents)
Telcom_data$PhoneService<-as.factor(Telcom_data$PhoneService)
Telcom_data$MultipleLines<-as.factor(Telcom_data$MultipleLines)
Telcom_data$InternetService<-as.factor(Telcom_data$InternetService)
Telcom_data$OnlineSecurity<-as.factor(Telcom_data$OnlineSecurity)
Telcom_data$OnlineBackup<-as.factor(Telcom_data$OnlineBackup)
Telcom_data$DeviceProtection<-as.factor(Telcom_data$DeviceProtection)
Telcom_data$TechSupport<-as.factor(Telcom_data$TechSupport)
Telcom_data$StreamingTV<-as.factor(Telcom_data$StreamingTV)
Telcom_data$StreamingMovies<-as.factor(Telcom_data$StreamingMovies)
Telcom_data$Contract<-as.factor(Telcom_data$Contract)
Telcom_data$PaperlessBilling<-as.factor(Telcom_data$PaperlessBilling)
Telcom_data$PaymentMethod<-as.factor(Telcom_data$PaymentMethod)
Telcom_data$Churn<-as.factor(Telcom_data$Churn)

#An alternate way of doing this using a function
#df[] <- lapply(df, function(x) if(is.character(x)) as.factor(x) else x)
#str(df)

#Re-checking the structure of the data set
str(Telcom_data)

#Missing value treatment - Eliminating the rows with missing values since there are very few missing observations when compared to the total number of observations in the data set.
missing_removed<-na.omit(Telcom_data)

#Checking the dimension of the data set after the elimination of missing values
dim(missing_removed)

#The data set now has 7032 observations (7032 customers) and 21 variables

#Checking the number of unique values after missing value removal
apply(missing_removed,2,function(x) length(unique(x)))

#Re-checking for missing values in the data set
sum(is.na(missing_removed))


#The data set now has no missing values

#The variable senior citizen has 0,1 instead of 'Yes','No'


#Converting the levels of 'Senior citizen' from 0&1 to 'Yes' & 'No'
missing_removed$SeniorCitizen<-ifelse(missing_removed$SeniorCitizen==0, 'No', 'Yes')


#Binning the 'tenure' variable

#The minimum value under 'tenure' is 1 month and maximum vaue is 72 months

missing_removed$Binned_tenure[missing_removed$tenure >=0 & missing_removed$tenure <= 12] <- '0-12 months'
missing_removed$Binned_tenure[missing_removed$tenure > 12 & missing_removed$tenure <= 24] <- '13-24 months'
missing_removed$Binned_tenure[missing_removed$tenure > 24 & missing_removed$tenure <= 36] <- '25-36 months'
missing_removed$Binned_tenure[missing_removed$tenure > 36 & missing_removed$tenure <= 48] <- '37-48 months'
missing_removed$Binned_tenure[missing_removed$tenure > 48 & missing_removed$tenure <= 60] <- '49-60 months'
missing_removed$Binned_tenure[missing_removed$tenure > 60 & missing_removed$tenure <= 72] <- '61-72 months'

View(missing_removed)

#The 3 continuous variables in the data set are of different units/scales. Hence standardizing them.

#Scaling the continuous variables using Z-score standardization

scaled_num<- missing_removed[,c("tenure", "MonthlyCharges", "TotalCharges")]
scaled_num<- data.frame(scale(scaled_num))

View(scaled_num)


#Preparing the final data set

#Dropping unwanted variables and attaching the scaled variables _drop only before modelling
missing_removed$tenure<-NULL
missing_removed$MonthlyCharges<-NULL
missing_removed$TotalCharges<-NULL

names(missing_removed)

#Eliminating the column 'Customer ID' since it will not help in further analysis
final_df$customerID<-NULL #Do not eliminate for EDA

final_df<-cbind(missing_removed,scaled_num)

names(final_df)
View(final_df)
dim(final_df)

#Exporting the pre-processed data
write.csv(final_df,"Pre-processed_telcom_VariablesNotEliminated.csv")
getwd()

#EXPLORATORY DATA ANALYSIS ON THE FINAL PROCESSED DATA

#Correlation among the numeric/continuous variables
corrdata<-subset(final_df,select=c("MonthlyCharges","tenure","TotalCharges"))

dim(corrdata)

correlation<-round(cor(corrdata),2)

correlation

#Re-shaping the correlation matrix to make a heat map

melted_cormat<-melt(correlation)
head(melted_cormat)

ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill=value))+geom_tile(color = "black")+scale_fill_gradient2(low = "blue", high= "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab",
                                                                                                 name="Correlation")
ggheatmap + geom_text(aes(Var2, Var1, label = value), color = "blue", size = 4)

#On an averge how many services have customers signed up for?

#Creating a list of all service names offered by the firm

services_only<-c("PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies")


#Creating a new column for the number of services that each customer has signed up for
final_data<-final_df %>%mutate(Number_of_services= rowSums(.[services_only] == "Yes"))
View(final_data)

#exporting final_data 
write.csv(final_data,"final_data_with_new_column.csv")

#Function for mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

getmode(final_data$Number_of_services)

mean(final_data$Number_of_services)
median(final_data$Number_of_services)

#What number of services have customers signed up for?
table(final_data$Number_of_services)

write.csv(final_data,"with_new_column.csv")

#How many customers have opted for all services
new_df<-final_data[(final_data$PhoneService == "Yes" & final_data$MultipleLines == "Yes" & final_data$OnlineSecurity == "Yes"& final_data$OnlineBackup == "Yes" & final_data$DeviceProtection == "Yes" & final_data$TechSupport== "Yes" & final_data$StreamingTV == "Yes" & final_data$StreamingMovies == "Yes"),]
dim(new_df)                    

#208 Customers have opted/signed up for all the available services. 

#Churn Vs No Churn among these 208 customers

table(new_df$Churn)

prop.table(table(new_df$Churn))*100

#Data frame of the churned customers among these 130

All_sevices_churn<-new_df[new_df$Churn=="Yes",]
dim(All_sevices_churn)
View(All_sevices_churn)

#Outlier detection for the continuous variables in the data set

t<-boxplot(missing_removed$tenure,main="Boxplot for Tenure",horizontal = TRUE,col = "orange",xlab="Tenure")
t$stats
t$out

m<-boxplot(missing_removed$MonthlyCharges,main="Boxplot for Monthly Charges",horizontal = TRUE,col = "red",xlab="Monthly Charges")
m$stats
m$out

to<-boxplot(missing_removed$TotalCharges,main="Boxplot for Total Charges",horizontal = TRUE,col = "purple",xlab="Total Charges")
to$stats
to$out

#Continuous variables against Churn (unscaled variables)

te<-boxplot(tenure~Churn,data=missing_removed,main="Tenure with Churn",xlab="Churn",ylab="Tenure",col=c("Orange","sky blue"))
te

#Customers who have churned after being associated with the firm for close to 6 years.

Churned<-missing_removed[missing_removed$Churn=="Yes",] #Customers who have churned
dim(Churned)

t_outliers<-Churned[(Churned$tenure==70|Churned$tenure==71|Churned$tenure==72),] #filtering customers who have tenure >=70 months
dim(t_outliers)
View(t_outliers)

mo<-boxplot(MonthlyCharges~Churn,data=missing_removed,main="Monthly Charges with Churn",xlab="Churn",ylab="Monthly Charges",col=c("Orange","sky blue"))
mo$stats

tot<-boxplot(TotalCharges~Churn,data=missing_removed,main="Total Charges with Churn",xlab="Churn",ylab="Total Charges",col=c("Orange","sky blue"))
tot

tot_outliers<-Churned[(Churned$TotalCharges>5637),] #filtering customers who have monthly charges>5637 (5638 is the smallest outlier)
dim(tot_outliers)
View(tot_outliers)

#Exploring t_outliers - customers who churned after 6 or more years
table(t_outliers$gender)
table(t_outliers$SeniorCitizen)
table(t_outliers$Partner)
table(t_outliers$Dependents)
table(t_outliers$PhoneService)
table(t_outliers$MultipleLines)
table(t_outliers$InternetService)
table(t_outliers$OnlineSecurity)
table(t_outliers$OnlineBackup)
table(t_outliers$DeviceProtection)
table(t_outliers$TechSupport)
table(t_outliers$StreamingTV)
table(t_outliers$StreamingMovies)
table(t_outliers$Contract)
table(t_outliers$PaperlessBilling)
table(t_outliers$PaymentMethod)

#Exploring tot_outliers
table(tot_outliers$gender)
table(tot_outliers$SeniorCitizen)
table(tot_outliers$Partner)
table(tot_outliers$Dependents)
table(tot_outliers$PhoneService)
table(tot_outliers$MultipleLines)
table(tot_outliers$InternetService)
table(tot_outliers$OnlineSecurity)
table(tot_outliers$OnlineBackup)
table(tot_outliers$DeviceProtection)
table(tot_outliers$TechSupport)
table(tot_outliers$StreamingTV)
table(tot_outliers$StreamingMovies)
table(tot_outliers$Contract)
table(tot_outliers$PaperlessBilling)
table(tot_outliers$PaymentMethod)


#On an average, how many services have these 23 customers signed up for?

#Exploring the two major services being offered - Phone Vs Internet

#Only phone
Only_phone<-subset(final_data,final_data$PhoneService=="Yes" & final_data$InternetService == "No")
dim(Only_phone)

(1520/7032)*100

#Churn - only phone
table(Only_phone$Churn)
prop.table(table(Only_phone$Churn))*100

#Only Internet
Only_internet<-subset(final_data,final_data$PhoneService=="No")
dim(Only_internet)

(680/7032)*100

#Churn - only phone
table(Only_internet$Churn)
prop.table(table(Only_internet$Churn))*100

#Both phone and internet
Both<-subset(final_data,final_data$PhoneService=="Yes" & final_data$InternetService == "DSL"|final_data$InternetService == "Fiber optic")
dim(Both)

(4832/7032)*100


#Checking if the data is balanced
barplot(table(final_df$Churn),col=c("maroon","blue"),main = "Data class distribution",xlab = 'Dependent variable - Churn',ylab = "count")

table(final_df$Churn)

#Clearly the data set is imbalanced.There are more 'NOs' in the data set than 'YESs'.

#Modelling

#Importing the pre-processed data
preprocessed<-read.csv(choose.files(),header = TRUE)

prop.table(table(preprocessed$Churn))

str(preprocessed)

#Converting all character variables to factor
preprocessed$gender<-as.factor(preprocessed$gender)
preprocessed$SeniorCitizen<-as.factor(preprocessed$SeniorCitizen)
preprocessed$Partner<-as.factor(preprocessed$Partner)
preprocessed$Dependents<-as.factor(preprocessed$Dependents)
preprocessed$PhoneService<-as.factor(preprocessed$PhoneService)
preprocessed$MultipleLines<-as.factor(preprocessed$MultipleLines)
preprocessed$InternetService<-as.factor(preprocessed$InternetService)
preprocessed$OnlineSecurity<-as.factor(preprocessed$OnlineSecurity)
preprocessed$OnlineBackup<-as.factor(preprocessed$OnlineBackup)
preprocessed$DeviceProtection<-as.factor(preprocessed$DeviceProtection)
preprocessed$TechSupport<-as.factor(preprocessed$TechSupport)
preprocessed$StreamingTV<-as.factor(preprocessed$StreamingTV)
preprocessed$StreamingMovies<-as.factor(preprocessed$StreamingMovies)
preprocessed$Contract<-as.factor(preprocessed$Contract)
preprocessed$PaperlessBilling<-as.factor(preprocessed$PaperlessBilling)
preprocessed$PaymentMethod<-as.factor(preprocessed$PaymentMethod)
preprocessed$Churn<-as.factor(preprocessed$Churn)
preprocessed$Binned_tenure<-as.factor(preprocessed$Binned_tenure)
preprocessed$Number_of_services<-as.factor(preprocessed$Number_of_services)

prop.table(table(preprocessed$Churn))

#Splitting the data into a train and test set in a 70:30 ratio

set.seed(1234)
s<-sample(1:nrow(preprocessed),0.7*nrow(preprocessed))

train<-preprocessed[s,]
dim(train)

test<-preprocessed[-s,]
dim(test)

table(train$Churn)

#FFTrees algorithm - To decide which classification algorithms are best for this data set

#Creating a logical label
train$Churn<-ifelse(train$Churn=="Yes", 'TRUE', 'FALSE')
test$Churn<-ifelse(test$Churn=="Yes", 'TRUE', 'FALSE')

table(test$Churn)
class(train$Churn)

train$Churn<-as.logical(train$Churn)
test$Churn<-as.logical(test$Churn)

# Create an FFTrees object 
churn.fft <- FFTrees(formula = Churn ~.,data = train,data.test = test,decision.labels = c("No", "Yes"))

#the print method shows aggregatge statistics
churn.fft

# Plot the best tree applied to the test data
plot(churn.fft,data = "test",main = "Customer Churn")

# Compare results across algorithms in test data
churn.fft$competition$test

#Calculating the F1 score for all algorithms

#FFTrees
(2 * (0.4774 * 0.8253) / (0.4774 + 0.8253))*100

#LR
(2 * (0.6727 * 0.5539) / (0.6727 + 0.5539))*100

#DT
(2 * (0.6397 * 0.5149) / (0.6397 + 0.5149))*100

#RF
(2 * (0.6465 * 0.5167) / (0.6465 + 0.5167))*100

#SVM
(2 * (0.6785 * 0.4628) / (0.6785 + 0.4628))*100

#Building the Logistic model on the unbalanced data
log_unbalanced<-glm(Churn~.,data = train,family = "binomial")

#Predicting the model on train data
p_train1 <- predict(log_unbalanced, train, type = 'response')
pred_tr1 <- ifelse (p_train1>0.5, "Yes","No")

#F1 Score train
F1_Score(train$Churn,pred_tr1,positive = "Yes")

#Predicting on the test data
p_test1 <- predict(log_unbalanced, test, type = 'response')
pred1 <- ifelse (p_test1>0.5, "Yes","No")

#Evaluating the model
confusionMatrix(as.factor(pred1),test$Churn, positive = 'Yes')

#F1 Score
F1_Score(test$Churn,pred1,positive = "Yes")

#Area under the curve
y1<-prediction(predictions=p_test1,labels=test$Churn)
y2<-performance(y1,"tpr","fpr") #true positive rate and false positive rate
plot(y2)

auc<-performance(y1,"auc")
as.numeric(auc@y.values) 

#Treating the data imbalance

#Random over sampling - Duplication of existing records with replacement
over <- ovun.sample(Churn~., data = train, method = "over", N = 7182)$data #N is the total number of observations that we want in the balanced data set.
#We have 3591 observations of 0 in the train. We will need another similar number of 1's also and hence N = 2*3591=7182.

table(over$Churn)

#Logistic model on the over sampled data
log_over<-glm(Churn~.,data = over,family = "binomial")

#Predicting on the train data
p_train2 <- predict(log_over, train, type = 'response')
pred_tr2 <- ifelse (p_train2>0.5, "Yes","No")

#F1 score
F1_Score(train$Churn,pred_tr2,positive = "Yes")

#Predicting on the test data
p_test2 <- predict(log_over, test, type = 'response')
pred2 <- ifelse (p_test2>0.5, "Yes","No")

#Evaluating the model
confusionMatrix(as.factor(pred2),test$Churn, positive = 'Yes')

#F1 Score
F1_Score(test$Churn,pred2,positive = "Yes")

#Area under the curve
y1<-prediction(predictions=p_test2,labels=test$Churn)
y2<-performance(y1,"tpr","fpr") #true positive rate and false positive rate
plot(y2)

auc<-performance(y1,"auc")
as.numeric(auc@y.values) 

#Random under sampling - random deletion of records
under <- ovun.sample(Churn~., data = train, method = "under", N = 2662)$data #N = 2*1331 (the number of 1's)
table(under$Churn)

#Logistic model on the under sampled data
log_under<-glm(Churn~.,data = under,family = "binomial")

#Model summary
summary(log_under)

#Predicting on the train data
p_train3 <- predict(log_under, train, type = 'response')
pred_tr3 <- ifelse (p_train3>0.5, "Yes","No")

#F1 score
F1_Score(train$Churn,pred_tr3,positive = "Yes")

#Predicting on the test data
p_test3 <- predict(log_under, test, type = 'response')
pred3 <- ifelse (p_test3>0.5, "Yes","No")

#Evaluating the model
confusionMatrix(as.factor(pred3),test$Churn, positive = 'Yes')

#F1 Score
F1_Score(test$Churn,pred3,positive = "Yes")

#Area under the curve
y1<-prediction(predictions=p_test3,labels=test$Churn)
y2<-performance(y1,"tpr","fpr") #true positive rate and false positive rate
plot(y2)

auc<-performance(y1,"auc")
as.numeric(auc@y.values) 


#Combining both over and under sampling
both <- ovun.sample(Churn~., data = train, method = "both",p = 0.5, seed = 1234,N = 4922)$data #N= number of 0's + number of 1's
table(both$Churn) #it wont be equal. The total number of observations will be 4922 but both classes will have silghtly different number of observations.

#Logistic model on the over sampled data
log_both<-glm(Churn~.,data = both,family = "binomial")

#Predicting on the train data
p_train4 <- predict(log_both, train, type = 'response')
pred_tr4 <- ifelse (p_train4>0.5, "Yes","No")

#F1 score
F1_Score(train$Churn,pred_tr4,positive = "Yes")

#Predicting on the test data
p_test4 <- predict(log_both, test, type = 'response')
pred4 <- ifelse (p_test4>0.5, "Yes","No")

#Evaluating the model
confusionMatrix(as.factor(pred4),test$Churn, positive = 'Yes')

#F1 Score
F1_Score(test$Churn,pred4,positive = "Yes")

#Area under the curve
y1<-prediction(predictions=p_test4,labels=test$Churn)
y2<-performance(y1,"tpr","fpr") #true positive rate and false positive rate
plot(y2)

auc<-performance(y1,"auc")
as.numeric(auc@y.values) 


#SMOTE
smoted_data <- SMOTE(Churn~., train, perc.over=100)
table(smoted_data$Churn)

#Logistic model on the over sampled data
log_smote<-glm(Churn~.,data = smoted_data,family = "binomial")

#Predicting on the train data
p_train5 <- predict(log_smote, train, type = 'response')
pred_tr5 <- ifelse (p_train5>0.5, "Yes","No")

#F1 score
F1_Score(train$Churn,pred_tr5,positive = "Yes")

#Predicting on the test data
p_test5 <- predict(log_smote, test, type = 'response')
pred5 <- ifelse (p_test5>0.5, "Yes","No")

#Evaluating the model
confusionMatrix(as.factor(pred5),test$Churn, positive = 'Yes')

#F1 Score
F1_Score(test$Churn,pred5,positive = "Yes")

#Area under the curve
y1<-prediction(predictions=p_test5,labels=test$Churn)
y2<-performance(y1,"tpr","fpr") #true positive rate and false positive rate
plot(y2)

auc<-performance(y1,"auc")
as.numeric(auc@y.values) 

#MODEL- 2 DECISION TREE

#Decision tree on the unbalanced data

DT_unbalanced<-rpart(Churn~.,data=train,method="class") #class means classification
DT_unbalanced

fancyRpartPlot(DT_unbalanced,type=1)

#Predicting on the train data
pred_dt_tr_1<-predict(DT_unbalanced,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_dt_tr_1,positive = "Yes")

#Predicting on the test data
pred_dt_1<-predict(DT_unbalanced,test,type = "class")
pred_dt_1

#Confusion matrix
confusionMatrix(as.factor(pred_dt_1),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_dt_1,positive = "Yes")


#Random over sampling
DT_over<-rpart(Churn~.,data=over,method="class")

fancyRpartPlot(DT_over,type=1)

#Predicting on the train data
pred_dt_tr_2<-predict(DT_over,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_dt_tr_2,positive = "Yes")

#Predicting on the test data
pred_dt_2<-predict(DT_over,test,type = "class")
pred_dt_2

#Confusion matrix
confusionMatrix(as.factor(pred_dt_2),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_dt_2,positive = "Yes")


#Random under sampling
DT_under<-rpart(Churn~.,data=under,method="class")

fancyRpartPlot(DT_under,type=1)

#Predicting on the train data
pred_dt_tr_3<-predict(DT_under,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_dt_tr_3,positive = "Yes")

#Predicting on the test data
pred_dt_3<-predict(DT_under,test,type = "class")
pred_dt_3

#Confusion matrix
confusionMatrix(as.factor(pred_dt_3),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_dt_3,positive = "Yes")


#Combining random over and under sampling
DT_both<-rpart(Churn~.,data=both,method="class")

fancyRpartPlot(DT_both,type=1)

#Predicting on the train data
pred_dt_tr_4<-predict(DT_both,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_dt_tr_4,positive = "Yes")

#Predicting on the test data
pred_dt_4<-predict(DT_both,test,type = "class")
pred_dt_4

#Confusion matrix
confusionMatrix(as.factor(pred_dt_4),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_dt_4,positive = "Yes")


#SMOTE
DT_smote<-rpart(Churn~.,data=smoted_data,method="class")

fancyRpartPlot(DT_smote,type=1)

#Predicting on the train data
pred_dt_tr_5<-predict(DT_smote,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_dt_tr_5,positive = "Yes")

#Predicting on the test data
pred_dt_5<-predict(DT_smote,test,type = "class")
pred_dt_5

#Confusion matrix
confusionMatrix(as.factor(pred_dt_5),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_dt_5,positive = "Yes")


#MODEL - 3 RANDOM FOREST

#Unbalanced
RF_unbalanced <- randomForest(Churn~.,data=train)

#Predicting on train data
pred_rf_tr_1<-predict(RF_unbalanced,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_rf_tr_1,positive = "Yes")

pred_rf_1<-predict(RF_unbalanced,test,type = "class")

#Confusion matrix
confusionMatrix(as.factor(pred_rf_1),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_rf_1,positive = "Yes")


#Random over sampling
RF_over <- randomForest(Churn~.,data=over)

#Predicting on train data
pred_rf_tr_2<-predict(RF_over,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_rf_tr_2,positive = "Yes")

pred_rf_2<-predict(RF_over,test,type = "class")

#Confusion matrix
confusionMatrix(as.factor(pred_rf_2),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_rf_2,positive = "Yes")


#Random under sampling

RF_under <- randomForest(Churn~.,data=under)

#Predicting on train data
pred_rf_tr_3<-predict(RF_under,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_rf_tr_3,positive = "Yes")

pred_rf_3<-predict(RF_under,test,type = "class")

#Confusion matrix
confusionMatrix(as.factor(pred_rf_3),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_rf_3,positive = "Yes")


#Combining over and under sampling

RF_both <- randomForest(Churn~.,data=both)

#Predicting on train data
pred_rf_tr_4<-predict(RF_both,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_rf_tr_4,positive = "Yes")

pred_rf_4<-predict(RF_both,test,type = "class")

#Confusion matrix
confusionMatrix(as.factor(pred_rf_4),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_rf_4,positive = "Yes")


#SMOTE

RF_smote <- randomForest(Churn~.,data=smoted_data)

#Predicting on train data
pred_rf_tr_5<-predict(RF_smote,train,type = "class")

#F1 score
F1_Score(train$Churn,pred_rf_tr_5,positive = "Yes")

pred_rf_5<-predict(RF_smote,test,type = "class")

#Confusion matrix
confusionMatrix(as.factor(pred_rf_5),test$Churn, positive = 'Yes')

#F1 score
F1_Score(test$Churn,pred_rf_5,positive = "Yes")


#The Best Model - Logistic Regression on Undersampled data (By variance in F1 scores)

#Building the step model to get rid of the insignificant variables

stepmodel<-step(log_under)

#Model summary

summary(stepmodel)



#SURVIVAL ANALYSIS

#Data for survival analysis
names(missing_removed)

#In the data set Churn = Yes: Churned , Churn = No: Censored

surv_data<-missing_removed #[,c(6,21)]

#Converting all variables to factor
surv_data[] <- lapply(surv_data, function(x) if(is.character(x)) as.factor(x) else x)
str(surv_data)

head(surv_data)

#KAPLAN MEIER METHOD

#creating the survival object
surv_data$survival<-Surv(surv_data$tenure,surv_data$Churn=="Yes")

#creating the life table - survival curve for the entire data set
lt<-survfit(survival ~ 1, data = surv_data) # 1 indiates a single survival curve.
lt

summary(lt)

#Kaplan Meier curve
ggsurvplot(lt, xlab = "Tenure",ylab = "Survival probability",risk.table = TRUE,censor=FALSE) #censor is used to keep or eliminate the lines

#Survival probability for the next 1 year
summary(lt,times=12)

#for the next 6 months
summary(lt,times=6)

#Comparing survival probabilities between groups
#Internet service
Int<-survfit(survival ~ InternetService, data = surv_data)
ggsurvplot(Int,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Multiple lines
ml<-survfit(survival ~ MultipleLines, data = surv_data)
ggsurvplot(ml,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Device protection
dp<-survfit(survival ~ DeviceProtection, data = surv_data)
ggsurvplot(dp,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Streaming tv
st<-survfit(survival ~ StreamingTV, data = surv_data)
ggsurvplot(st,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Streaming movies
sm<-survfit(survival ~ StreamingMovies, data = surv_data)
ggsurvplot(sm,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Contract
c<-survfit(survival ~ Contract, data = surv_data)
ggsurvplot(c,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Paperless billing
pb<-survfit(survival ~ PaperlessBilling, data = surv_data)
ggsurvplot(pb,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Payment method
pm<-survfit(survival ~ PaymentMethod, data = surv_data)
ggsurvplot(pm,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)

#Binned tenure
bt<-survfit(survival ~ Binned_tenure, data = surv_data)
ggsurvplot(bt,censor=FALSE,xlab = "Tenure",ylab = "Survival probability",pval = TRUE)


#Cox proportional hazard model

#Building the cox proportional hazard model on the significant variables (returned by logistic/step model) from the data set

#Preparing the data set by retaining only the significant variables,the dependent variable and the tenure variable(since tenure and Churn are required to create the survival object)

sig_vars<-missing_removed[,c("Churn","MultipleLines","InternetService","DeviceProtection","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Binned_tenure","MonthlyCharges","tenure")]
names(sig_vars)
dim(sig_vars)

#Creating the survival object
sig_vars$survival<-Surv(sig_vars$tenure,sig_vars$Churn=="Yes")

#Fitting the Cox proportional hazards model
fit.coxph <- coxph(survival ~ MultipleLines + InternetService + DeviceProtection + StreamingTV + StreamingMovies + Contract + PaperlessBilling + PaymentMethod + Binned_tenure + MonthlyCharges,data = sig_vars)
summary(fit.coxph)

#ggforest plot
ggforest(fit.coxph, data = sig_vars)
