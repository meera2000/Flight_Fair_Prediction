#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Part-1: data Exploration and Pre-processing

# In[6]:


cd D:\FT\python\Ml\Assignment


# In[7]:


# 1) load the given dataset 
data=pd.read_csv("Python_Project_9_Ran.csv")


# In[8]:


data.head()


# In[12]:


# 2) Print the unique values in all column
for col in data.columns:
    print(col,data[col].unique())


# In[13]:


# 3) Fill nan value with ‘other’
data['country'].fillna('country',inplace=True)


# In[16]:


data['country'].isnull().sum()


# In[18]:


# 4) Fill nan in agent with mean of agent columns
data['agent'].fillna(data['agent'].mean(),inplace=True)


# In[19]:


data['agent'].isnull().sum()


# In[24]:


data.isnull().sum()


# In[23]:


# 5) Drop all the remaining null value
data.dropna(inplace=True)


# In[38]:


data['adults'].value_counts().index


# In[33]:


# 6) Plot the count of adult and children with help of a bar plot
data['adults'].value_counts().plot(kind = 'bar')


# In[39]:


plt.bar(data['adults'].value_counts().index,data['adults'].value_counts().values)


# In[40]:


plt.bar(data['children'].value_counts().index,data['children'].value_counts().values)


# In[42]:


data = data.drop('reservation_status_date' , axis = 1 )   


# In[41]:


# 7) Perform Label encoding on categorical columns
from sklearn.preprocessing import LabelEncoder


# In[43]:


enc=LabelEncoder()


# In[46]:


data['hotel'] = enc.fit_transform(data['hotel'])
data['country'] = enc.fit_transform(data['country'])
data['hotel'] = enc.fit_transform(data['hotel'])
data['market_segment'] = enc.fit_transform(data['market_segment'])
data['distribution_channel'] = enc.fit_transform(data['distribution_channel'])
data['meal'] = enc.fit_transform(data['meal'])
data['reserved_room_type'] = enc.fit_transform(data['reserved_room_type'])
data['assigned_room_type'] = enc.fit_transform(data['assigned_room_type'])
data['deposit_type'] = enc.fit_transform(data['deposit_type'])
data['customer_type'] = enc.fit_transform(data['customer_type'])
data['reservation_status'] = enc.fit_transform(data['reservation_status'])
data['arrival_date_month'] = enc.fit_transform(data['arrival_date_month'])


# In[47]:


data.head()


# In[48]:


data.info()


# In[49]:


# Part 2 Model Building
# 1. Create features and target data
x=data.drop('is_canceled',axis=1)
y=data.is_canceled


# In[50]:


# 2. Split into training & testing
from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[52]:


# 3. Apply Random forest classifier on data
from sklearn.ensemble import RandomForestClassifier


# In[54]:


model=RandomForestClassifier()
model.fit(x_train,y_train)


# In[57]:


pred=model.predict(x_test)


# In[58]:


pred


# In[68]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,accuracy_score


# In[74]:


# 4. Create function which show Precision score, recall score, accuracy, classification report and confusion matrix
def fun():
        print(precision_score(y_test,pred))
        print(recall_score(y_test, pred))
        print(accuracy_score(y_test, pred))
        print(classification_report(y_test,pred,digits=5))
        print(confusion_matrix(y_test,pred))


# In[75]:


fun()

