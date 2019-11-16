#Importing required libraries
import pandas as pd
import keras as ke
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import datetime  as dt
from dateutil.relativedelta import relativedelta
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

#Reading seattle dataset
filename = "C://Users//14694//Documents//Education//Project//Seattle_Police_Department_911_Incident_Response.csv"
df = pd.read_csv(filename,parse_dates=["Event Clearance Date", "At Scene Time"],encoding='utf8')

#Drop null/empty/missing values
df = df.dropna()

#Changing data types
df[["CAD CDW ID"]] = df[["CAD CDW ID"]].astype(int)

obj_df =df.select_dtypes(include=['object']).copy()
for col in obj_df:
    obj_df[col] = obj_df[col].astype('category')
df[obj_df.columns]=obj_df[obj_df.columns]

#Reducing the rarely occured(less than 5%) categorical levels 
cat_col = df[['Initial Type Group','Event Clearance Group','District/Sector']]
for i in cat_col:
    group_count = pd.value_counts(df[i])
    freq = (group_count/group_count.sum() * 100).lt(5)
    df[i] = np.where(df[i].isin(group_count[freq].index),'Other',df[i])
    print(df[i].nunique())
    
#Calculating TIme duration from "At Scene Time" and "Event Clearance Date"
df['Time Duration'] = df['Event Clearance Date'] - df['At Scene Time']
df['Time Duration'] = df['Time Duration']/np.timedelta64(1,'h')

#Removing invalid time values(negative values)
df = df[df['Time Duration'] > 0]

#Visualize the time spent on different events over all district/sectors in Seattle
df_visulaize = df[['Time Duration','District/Sector', 'Event Clearance Group']]
heatmap_data = pd.pivot_table(df_visulaize,values='Time Duration', index=['Event Clearance Group'], columns='District/Sector')
sns.heatmap(heatmap_data, cmap="BuGn")

#Dropping Insignificant features
to_drop = ["Initial Type Description", "Event Clearance Description","Initial Type Subgroup", 'Event Clearance SubGroup', 'Hundred Block Location',"Incident Location","Zone/Beat",'Census Tract','Event Clearance Group','Event Clearance Code','Event Clearance Date','Longitude','Latitude','Event Clearance Date','At Scene Time']
df.drop(df[to_drop], inplace=True, axis=1)

#Creating dummy variables for all categorical features
df_dummy =pd.get_dummies(df,drop_first=True)

#Plot correlation matix
plt.figure(figsize=(25,20))
corr = df_dummy[df_dummy.columns.difference(['CAD CDW ID','CAD Event Number','General Offense Number'])].corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()

#Creating train(80%) and test data(20%)
train = df_dummy.sample(frac=0.8,random_state=100) #random state is a seed value
test = df_dummy.drop(train.index)

# Seperating the identity feauters(not adding any value to prediction)
df_CDWIDs_train = train[['CAD CDW ID', 'CAD Event Number', 'General Offense Number']].copy()
df_train = train.drop(['CAD CDW ID', 'CAD Event Number', 'General Offense Number'], axis=1)
df_CDWIDs_test = test[['CAD CDW ID', 'CAD Event Number', 'General Offense Number']].copy()
df_test = test.drop(['CAD CDW ID', 'CAD Event Number', 'General Offense Number'], axis=1)


#Train linear regression model
model = LinearRegression()
train.model = model.fit(df_train.loc[:,df_train.columns != 'Time Duration'], df_train.loc[:,df_train.columns == 'Time Duration'])

#Predicting target variable of train data
pred_train = train.model.predict(df_train.loc[:,df_train.columns != 'Time Duration'])

#Predicting target variable of test data
pred_test = train.model.predict(df_test.loc[:,df_test.columns != 'Time Duration'])

#Measure regression metrics
print(metrics.mean_absolute_error(df_test.loc[:,df_test.columns == 'Time Duration'],pred_test))
print(metrics.mean_squared_error(df_test.loc[:,df_test.columns == 'Time Duration'],pred_test))
print(np.sqrt(metrics.mean_squared_error(df_test.loc[:,df_test.columns == 'Time Duration'],pred_test)))

#Save the model
joblib.dump(train.model,'SeattleModel.pkl')
