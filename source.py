#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing prediction Project

# Problem Statement:Predict the count  of bikes rented on hourly basis 

 

'''Step 0)  Analysis of data
   Step 1)  Reading of data
   Step 2)  Prelimary analysis of data dropping of obvious feature
                i) Ids and other non important features
   Step 3)  Dealing with null values
                i)  If the number of null values is more than 50% Drop that columns
                ii) If null value is int/float fill the data by mean of that columns
                iii)If null value is object/category fill the data by mode of that columns
                iv) Convert the data into its appropriate data types
   Step 4)  Visualise the data
                i)This helps in seeing the correlation of data so we can drop the fields which are highly correlated
                ii)How clean is our data
   Step 5)  Check for regression assumptions
                i)normality
                ii)autocorrealtion
                iii)correaltions
                iv)outliers 
   Step 6)  Drop irrelevant features
   Step 7)  Create/ modify features
   Step 8)  Feature engineering
   Step 9)  Create dummy variables(For categorical features)
   Step 10)  Train and test Split()
   Step 11) Fit and score the model
                i)Model Cross Validation
                ii)HyperparameterTuning 
   Step 12) Present the results'''




''''Step 0) Analysis OF data
- instant: record index
- dteday : date
- season : season (1:winter, 2:spring, 3:summer, 4:fall)
- yr : year (0: 2011, 1:2012)
- mnth : month ( 1 to 12)
- hr : hour (0 to 23)
- holiday : weather day is holiday or not (extracted from [Web Link])
- weekday : day of the week
- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
+ weathersit :
- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered'''


# Step 1) Reading of data and importing all the important header files




#importing all the imporant header file
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
plt.style.use('bmh')




#To read the csv file
df=pd.read_csv("hour.csv")
df




#To check the number of row and colums in the dataset
print(df.shape)




#to check the data types and null values of the field
df.info()



df.head()



'''Step 2)  Prelimary analysis of data dropping of obvious feature
                i) "instant" beacuse it is not impoartant
                ii)"Yr" cause we can take this field again from dte field
                iii)"mth" cause we can take this field again from dte field
                iv)"casual" cause  the problem statement says count not which one
                v)"registered" cause the problem statement says count not which one
'''                





df=df.drop(["instant","yr",'mnth','casual','registered'],axis=1)
df





df["dteday"]=pd.to_datetime(df["dteday"])
df.info()





''''  Step 3)  Dealing with null values
               i)  If the number of null values is more than 50% Drop that columns
               ii) If null value is int/float fill the data by mean of that columns
               iii)If null value is object/category fill the data by mode of that columns
           '''





#to check the fields which contains null values if any 
df.isnull().sum()



df["season"]=df["season"].astype("category")
df["hr"]=df["hr"].astype("category")
df["weekday"]=df["weekday"].astype("category")
df["workingday"]=df["workingday"].astype("category")
df["weathersit"]=df["weathersit"].astype("category")
df["holiday"]=df["holiday"].astype("category")



df.info()





''' Step 4)  Visualise the data
                i)This helps in seeing the correlation of data so we can drop the fields which are highly correlated
                ii)How clean is our data
'''




#continious variables visualization
c_features = list(df[['temp', 'atemp', 'hum', 'windspeed']].columns)
color = ['a','g','b','r','b','g','b','r','b']
sp = 1
for columns in c_features:
    plt.subplot(2,2,sp)
    plt.title(columns + ' Vs count')
    plt.xlabel(columns)
    plt.ylabel('count')
    plt.scatter(df[columns],df['cnt'], s=2, c=color[sp])
    sp+=1
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.show()



#categorical variables visualization
cg_features = list(df[['season','hr','holiday','weekday','workingday']].columns)
sp=1
for columns in cg_features:
    plt.subplot(2,4,sp)
    plt.title('Average count per ' + columns)
    cat_list = df[columns].unique()
    cat_average = df.groupby(columns).mean()['cnt']
    plt.bar(cat_list,cat_average, color = color[sp])
    sp+=1
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(20.5, 20.5)
plt.show()


# we will drop weeekday and working data because we dont see much difference and they are highly correlated 




df=df.drop(["weekday",'workingday'],axis=1)
df




'''  Step 5)  Check for regression assumptions
                i)normality
                ii)autocorrealtion
                iii)correaltions
                iv)outliers'''





df.corr()




df1=pd.to_numeric(df["cnt"],downcast="float")
plt.figure()
df1.hist(rwidth=0.9, bins=20)


# as you can see the figure is right skewed so we will use log function to convert the to normal distributions


df2=np.log(df1)
df2
plt.figure()
df2.hist(rwidth=0.9, bins=20)
plt.show()



#now we will check for autocorreation
plt.acorr(df1,maxlags=12)
plt.show()

# Step 7)  Create/ modify features


days=10
df["predict"]=df['cnt'].shift(-days)
df=df.dropna()




df2=pd.to_numeric(df["predict"],downcast='float')
df2
plt.acorr(df2,maxlags=12)


# step 9) Create dummy varibles 

df_dumb=pd.get_dummies(df)