rm(list=ls(all=TRUE))
gc(reset=TRUE)

packages<-c("data.table", "caret", "corrplot", "reshape2", "ggplot2",
            "mlr","DMwR","psych","Matrix","xgboost","e1071","randomForest",
            "MASS","class","nnet")

install.packages(packages)

library("data.table")
library("caret")
library("corrplot")
library("reshape2")
library("ggplot2")
library("psych")
library("mlr")
library("DMwR")
library("Matrix")
library("xgboost")
library("e1071")
library("randomForest")
library("MASS")
library("nnet")
library("class")

options(scipen = 999) #remove exponential annotation

# loading Train-Test datasets without ID and Dates variables

train <- fread("https://s3-eu-west-1.amazonaws.com/uploads-eu.hipchat.com/13008/2786591/oMCJMoMERdb4ZvJ/train_set_3weeks.csv",
                        drop=c(1:2,17:18),na.strings=c("","-","-,-"))

test <- fread("https://s3-eu-west-1.amazonaws.com/uploads-eu.hipchat.com/13008/2786591/na2J2QUhaYEUf76/test_4weeks.csv",
               drop=c(1:2,17:18),na.strings=c("","-","-,-"))

head(train)
head(test)

# renaming Response variable
colnames(train)[19]<-c("card_scheme")
colnames(test)[19]<-c("card_scheme")

# checking for unique values
sapply(train, function(x) length(unique(x)))
sapply(test, function(x) length(unique(x)))

# turning into factors
factcols<-c(1:3,7:17,19)
train[,(factcols) := lapply(.SD,factor), .SDcols=factcols]
test[,(factcols) := lapply(.SD,factor), .SDcols=factcols]

# turning into numerics
numcols <- setdiff(1:19,factcols)
train[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]
str(train)

test[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]
str(test)

# dropping unused factor levels
train<-droplevels(train)
test<-droplevels(test)

# checking for NAs
sum(is.na(train))
sum(is.na(test))

# NA percentage for each feature
(mis.val.train <- sapply(train, function(x){sum(is.na(x))/length(x)})*100)
(mis.val.test <- sapply(test, function(x){sum(is.na(x))/length(x)})*100)

# removing NAs
train<-na.omit(train)
sum(is.na(train))

test<-na.omit(test)
sum(is.na(test))

# creating train-test datasets just with factorial features
factors.train <- train[,factcols, with=FALSE]
factors.test <- test[,factcols, with=FALSE]

# creating train-test datasets just with numerical features
num.train <- train[,numcols,with=FALSE]
num.test <- test[,numcols,with=FALSE]

# checking for 0 variance variables

nearZeroVar(train,saveMetrics = TRUE)

# checking correlations between numerics

png(filename="corrplot.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)

corrplot.mixed(cor(num.train),lower="circle", 
               upper="color",tl.pos="lt", diag="n", order="hclust", 
               hclust.method="complete")
dev.off()

# plotting numerics hists

melt.train<-melt(num.train)

png(filename="hist1.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)

ggplot(data = melt.train, aes(x = value)) +
  geom_histogram(bins =15) + geom_density()+
  facet_wrap(~variable, scales = "free")
dev.off()

str(factors.train)
str(factors.test)

# recoding factor levels by keeping less (oles oses me levels < 53)
factcols.b<-c(2,5,8:12)
factors.train.b <- factors.train[,factcols.b, with=FALSE]

for(i in names(factors.train.b)){
  p <- 1/100
  ld <- names(which(prop.table(table(factors.train.b[[i]])) < p))
  levels(factors.train.b[[i]])[levels(factors.train.b[[i]]) %in% ld] <- "Other"
}

str(factors.train.b)

factors.test.b <- factors.test[,factcols.b, with=FALSE]

for(i in names(factors.test.b)){
  p <- 1/100
  ld <- names(which(prop.table(table(factors.test.b[[i]])) < p))
  levels(factors.test.b[[i]])[levels(factors.test.b[[i]]) %in% ld] <- "Other"
}

str(factors.test.b)

# creating new.train - new.test datasets with new factor levels
factcols<-setdiff(1:15,factcols.b)
factors.train <- factors.train[,factcols, with=FALSE]
factors.train<-cbind(factors.train.b,factors.train)
rm(factors.train.b,numcols,factcols,melt.train,mis.val.train,i,ld,p)

new.train<-cbind(num.train,factors.train)
str(new.train)

rm(num.train,factors.train,train) #to save memory

factcols<-setdiff(1:15,factcols.b)
factors.test <- factors.test[,factcols, with=FALSE]
factors.test<-cbind(factors.test.b,factors.test)
rm(factors.test.b,factcols,factcols.b,mis.val.test)

new.test<-cbind(num.test,factors.test)
str(new.test)

rm(num.test,factors.test,test) #to save memory

# get new descriptives

describe(new.train)
describe(new.test)

# log transformation for skewed variables

new.train$children<-log10(new.train$children+1)
new.test$children<-log10(new.test$children+1)

new.train$infants<-log10(new.train$infants+1)
new.test$infants<-log10(new.test$infants+1)

new.train$Sales<-log10(new.train$Sales+1)
new.test$Sales<-log10(new.test$Sales+1)

melt.train<-melt(new.train)


png(filename="hist2.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)

ggplot(data = melt.train, aes(x = value)) +
  geom_histogram(bins=15) +
  facet_wrap(~variable, scales = "free")
dev.off()

# get descriptives

describe(new.train)
describe(new.test)

# checking number of levels of each factor in each dataset

summarizeColumns(new.train)[,"nlevs"]
summarizeColumns(new.test)[,"nlevs"]

# merging factor levels

tmp <- union(levels(new.train$country_map),
             levels(new.test$country_map))

levels(new.train$country_map) <- tmp
levels(new.test$country_map) <- tmp


tmp <- union(levels(new.train$affiliate_id),
             levels(new.test$affiliate_id))

levels(new.train$affiliate_id) <- tmp
levels(new.test$affiliate_id) <- tmp

tmp <- union(levels(new.train$origin_airport),
             levels(new.test$origin_airport))

levels(new.train$origin_airport) <- tmp
levels(new.test$origin_airport) <- tmp

tmp <- union(levels(new.train$destination_airport),
             levels(new.test$destination_airport))

levels(new.train$destination_airport) <- tmp
levels(new.test$destination_airport) <- tmp

tmp <- union(levels(new.train$fare_classes),
             levels(new.test$fare_classes))

levels(new.train$fare_classes) <- tmp
levels(new.test$fare_classes) <- tmp

tmp <- union(levels(new.train$marketing_carriers),
             levels(new.test$marketing_carriers))

levels(new.train$marketing_carriers) <- tmp
levels(new.test$marketing_carriers) <- tmp

tmp <- union(levels(new.train$currency_map),
             levels(new.test$currency_map))

levels(new.train$currency_map) <- tmp
levels(new.test$currency_map) <- tmp

tmp <- union(levels(new.train$language_map),
             levels(new.test$language_map))

levels(new.train$language_map) <- tmp
levels(new.test$language_map) <- tmp

rm(tmp,melt.train)

# recheck of 0 variance

nearZeroVar(new.train,saveMetrics = TRUE)

# drop 'full_route' feature due to 0 variance

new.train<-new.train[,full_route:=NULL]
new.test<-new.test[,full_route:=NULL]

# recoding response levels for faster computing

num.class<-length(levels(new.train$card_scheme))
levels(new.train$card_scheme)<-1:num.class
levels(new.test$card_scheme)<-1:num.class
rm(num.class)

# handling Imbalanced response using SMOTE technique
set.seed(666)

table(new.train$card_scheme)

new.train<-SMOTE(card_scheme ~ ., data  = new.train, perc.over = 36000,
                      perc.under=590,k=3)                         
table(new.train$card_scheme) 


table(new.test$card_scheme)

new.test<-SMOTE(card_scheme ~ ., data  = new.test, perc.over = 50000,
                      perc.under=450,k=3)  
table(new.test$card_scheme) 

# turn the data tables into sparse matrices and create label variable

sp.mat.train<-sparse.model.matrix(card_scheme~.-1,data=new.train)
sp.mat.test<-sparse.model.matrix(card_scheme~.-1,data=new.test)

## xGBoost classifier & confusion matrix & accuracy
set.seed(666)

label<-as.numeric(unlist(new.train[,"card_scheme",with=FALSE]))

param<-list("objective" = "multi:softmax",
              "num_class" = 8,
              "eval_metric" = "merror")


xgb.model<-xgboost(param=param, data=sp.mat.train, label=label,nrounds=100,
                  verbose=TRUE)

xgb.pred<-predict(xgb.model, sp.mat.test) 

importance <- xgb.importance(feature_names = colnames(sp.mat.train), 
                             model = xgb.model)

png(filename="xgb-importance.png", 
    type="cairo",
    units="in", 
    width=15, 
    height=6, 
    pointsize=12, 
    res=150)

xgb.plot.importance(importance,top_n=10)
dev.off()

(cm<-confusionMatrix(new.test$card_scheme,xgb.pred))

cm$byClass

## SVM classifier & confusion matrix & accuracy
set.seed(666)

svm.model<-svm(sp.mat.train,new.train$card_scheme,kernel = "radial",cost=1,gamma=0.0045)

svm.model

svm.pred<-predict(svm.model,sp.mat.test)

(svm.cm<-confusionMatrix(new.test$card_scheme,svm.pred))

svm.cm$byClass

## RF classifier & confusion matrix & accuracy
set.seed(666)

rf.model<-randomForest(card_scheme~.,data=new.train,ntree=500,mtry=4,
                       importance=TRUE)

rf.pred<-predict(rf.model,new.test,type="response")

(rf.cm<-confusionMatrix(new.test$card_scheme,rf.pred))

rf.cm$byClass

varImp(rf.model)

windows()
varImpPlot(rf.model,type=2)


## LDA classifier & confusion matrix & accuracy
set.seed(666)

num.train<-new.train[,c(1:4,18),with=FALSE]

num.test<-new.test[,c(1:4,18),with=FALSE]

lda.model<-lda(card_scheme~.,data=new.train)


lda.pred<-predict(lda.model,new.test,type="response")

(lda.cm<-confusionMatrix(new.test$card_scheme,lda.pred$class))

lda.cm$byClass

## NB classifier & confusion matrix & accuracy
set.seed(666)

nb.model<-naiveBayes(card_scheme~.,data=new.train)

nb.pred<-predict(nb.model,new.test,type="class")

(nb.cm<-confusionMatrix(new.test$card_scheme,nb.pred))

nb.cm$byClass

## NNet classifier & confusion matrix & accuracy (!!!! ola st 3)
set.seed(666)

nnet.model<-nnet(card_scheme~.,data=new.train,size=10,maxit=1000,MaxNWts=2400)

nnet.pred<-predict(nnet.model,new.test,type="class")

(nnet.cm<-confusionMatrix(new.test$card_scheme,nnet.pred))

nnet.cm$byClass

## multinomial Logistic Regression classifier & confusion matrix & accuracy
set.seed(666)

mn.model<-multinom(card_scheme~.,data=new.train,maxit=200,MaxNWts=1600)

mn.pred<-predict(mn.model,new.test,type="class")

(mn.cm<-confusionMatrix(new.test$card_scheme,mn.pred))

mn.cm$byClass

## k-NN classifier & confusion matrix & accuracy
set.seed(666)

k<-round(sqrt(nrow(new.train)))

knn.pred<-knn(sp.mat.train,sp.mat.test,new.train$card_scheme,k=5)

(knn.cm<-confusionMatrix(new.test$card_scheme,knn.pred))

knn.cm$byClass
