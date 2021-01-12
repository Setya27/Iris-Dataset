#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[3]:


# LOAD DATASET
df = pd.read_csv('Iris.csv')
df.head()


# In[4]:


# CHECK NULL VALUE
df.isnull().sum()


# In[5]:


# COLUMNS IN DATA
df.columns


# In[6]:


# DATA SHAPE
df.shape


# In[7]:


# UNIQUE VALUES OF SPECIES
df['Species'].unique()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.groupby(by='Species').agg(['mean', 'min', 'max'])


# In[11]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm');


# In[12]:


sns.heatmap(df.corr(), annot=True, cmap='flag');


# In[13]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df);


# In[14]:


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df);


# In[15]:


sns.pairplot(df, hue='Species')


# In[16]:


X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[18]:


svc = SVC()
svc.fit(X_train, y_train)


# In[19]:


y_pred = svc.predict(X_test)


# In[20]:


print(confusion_matrix(y_test, y_pred))


# In[21]:


print(classification_report(y_test, y_pred))

