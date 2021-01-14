#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

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

# LOAD DATASET
df = pd.read_csv('Iris.csv')
df.head()

# CHECK NULL VALUE
df.isnull().sum()

# COLUMNS IN DATA
df.columns

# DATA SHAPE
df.shape

# UNIQUE VALUES OF SPECIES
df['Species'].unique()

# EDA
df.describe()
df.info()
df.groupby(by='Species').agg(['mean', 'min', 'max'])
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm');


# Dataset Visualization
sns.heatmap(df.corr(), annot=True, cmap='flag');

sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df);

sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df);

sns.pairplot(df, hue='Species')


# SVM (Support Vector Machine Classifier)
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

