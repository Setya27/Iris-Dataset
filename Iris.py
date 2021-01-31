#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('Iris.csv')


# In[4]:


df.describe()


# In[5]:


df.groupby('Species').describe().round(2)


# In[6]:


def box_plot_species(df, column):
    sns.boxplot(data=df, x='Species', y=column)


# In[7]:


box_plot_species(df, 'SepalLengthCm')


# In[8]:


box_plot_species(df, 'SepalWidthCm')


# In[9]:


box_plot_species(df, 'PetalLengthCm')


# In[10]:


box_plot_species(df, 'PetalWidthCm')


# In[11]:


f, ax = plt.subplots(figsize=(12,6))
a = sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species', ax=ax);
a.set_title('Scatter Plot SepalWidth and SepalLength')


# In[12]:


f, ax = plt.subplots(figsize=(12,6))
b = sns.scatterplot(data=df, x='PetalLengthCm', y='PetalWidthCm', hue='Species', ax=ax);
b.set_title('Scatter Plot PetalWidth and PetalLength');

