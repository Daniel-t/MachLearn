Distinguishing Correct Excercise Technique with the HAR dataset
===============================================================
## Daniel Thomas 2014
This report was written for the Practical Machine Learning Coursera class running July 2014.

In this activity we analyse data from http://groupware.les.inf.puc-rio.br/har which includes measurements from multiple sensors during a specific excercaise.  Each record is marked as to if the excercise technique is correct 'Classe A' or a specific incorrect technique 'Classe B-E'.   We take this data and attempt to constuct a machine learning algorithm which will correctly identify future samples to determine if the technique is correct, or otherwise which error is being made.


### Environment
We start by setting the environment, establishing a fixed seed (for reproducibility) and loading necessary libraries.

```r
library(plyr)
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:plyr':
## 
##     arrange, desc, failwith, id, mutate, summarise, summarize
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12321)
```
### Data load and cleanse
Data is loaded from the training file and the following transformations are made
. records with new_window == "yes" are discarded as these contain summary data which is not useful for this excercise
. columns beginning with stddev, var, avg, max, min, amplitude, kurtosis and skewness are discarded. These only have a value for new_window=="yes" records, so are not value adding here.
. Columns X, username, timestamps (all three), new_window & num_window are discarded
. the 'classe' column is converted to a factor.

```r
allData<-read.csv(file="data//pml-training.csv",na.strings = "NA",stringsAsFactors=F)

#drop all new_window rows
allData<-allData %>% filter(new_window!="yes")
allData<-allData %>% select(-starts_with('stddev_'),-starts_with('var_'),-starts_with('avg_'),-starts_with('min_'),-starts_with('amplitude_'),-starts_with('max_'),-starts_with('kurtosis'),-starts_with('skewness'))

allData<-allData %>% select(-X,-user_name,-starts_with('raw_timestamp_part'),-cvtd_timestamp,-new_window,-num_window)

#convert classe to a factor

allData<-allData %>% mutate(classe=as.factor(classe))
```

### Training and Validation DataSets
After the initial transformations, data is split into a training and cross validation set, such that around 70% of records are avaialable for training, with the remainder used for checking for validation testing.


```r
inTrain<-createDataPartition(allData$classe,p=0.7,list=F)
training<-allData[inTrain,]
cvTrain<-allData[-inTrain,]
```

### Model building
Now we create a model from the training data set with the 'GBM' method.   Once the model is built we print the confusion matrix to show the level of accuracy obtained. 

```r
mod<-train(classe ~. ,data=training,method="gbm",verbose=F)
predTrain<-predict(mod,training)
confusionMatrix(training$classe,predTrain)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3794   23   10    2    1
##          B   59 2502   39    3    0
##          C    0   51 2276   17    3
##          D    0    8   44 2147    4
##          E    5   16   18   25 2406
## 
## Overall Statistics
##                                         
##                Accuracy : 0.976         
##                  95% CI : (0.973, 0.978)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.969         
##  Mcnemar's Test P-Value : 1.21e-14      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.983    0.962    0.953    0.979    0.997
## Specificity             0.996    0.991    0.994    0.995    0.994
## Pos Pred Value          0.991    0.961    0.970    0.975    0.974
## Neg Pred Value          0.993    0.991    0.990    0.996    0.999
## Prevalence              0.287    0.193    0.177    0.163    0.179
## Detection Rate          0.282    0.186    0.169    0.160    0.179
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.990    0.977    0.974    0.987    0.995
```
As can be seen with the training set an accuracy of **97.6%** was obtained, leaving an error rate of **2.4%**, the 95% confidence interval is 97.3% to 97.8% 

### Model Validation
Finally our model is validated against the data kept aside for validation and the confusion matrix supplied below.

```r
predTest<-predict(mod,cvTrain)
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
```

```r
confusionMatrix(cvTrain$classe,predTest)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1613   21    6    1    0
##          B   24 1065   25    1    0
##          C    0   28  960   16    1
##          D    3    6   25  907    3
##          E    2   13    9   15 1019
## 
## Overall Statistics
##                                       
##                Accuracy : 0.965       
##                  95% CI : (0.96, 0.97)
##     No Information Rate : 0.285       
##     P-Value [Acc > NIR] : < 2e-16     
##                                       
##                   Kappa : 0.956       
##  Mcnemar's Test P-Value : 6.59e-06    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.982    0.940    0.937    0.965    0.996
## Specificity             0.993    0.989    0.991    0.992    0.992
## Pos Pred Value          0.983    0.955    0.955    0.961    0.963
## Neg Pred Value          0.993    0.985    0.986    0.993    0.999
## Prevalence              0.285    0.197    0.178    0.163    0.178
## Detection Rate          0.280    0.185    0.167    0.157    0.177
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.988    0.965    0.964    0.979    0.994
```

This shows an accuracy of **96.5%**, or an error rate of **3.5%** which can be expected to be similar to new sample data as the validation data was not used in construction of the model.   The 95% Confidence Interval for the validation test is 96% to 97%.   This is a very good result, so some confidence can be placed on predictions made using this model.
