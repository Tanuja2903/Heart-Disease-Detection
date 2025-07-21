#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:\heart.csv")
df.head()


# In[4]:


df.isnull().sum()


# In[9]:


sns.scatterplot(x='age',y='chol',data=df,color='b')


# In[10]:


sns.barplot(x='cp',y='age',data=df)


# In[24]:


sns.barplot(x='output',y='cp',data=df)


# **Train Test Split

# In[25]:


x = df.drop(['output'],axis=1)
y = df['output']


# In[26]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[27]:


sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)


# **Logistic Regression

# In[28]:


lr = LogisticRegression()


# In[29]:


lr.fit(xtrain,ytrain)


# In[30]:


lr.score(xtest,ytest)


# In[31]:


yp = lr.predict(xtest)


# In[32]:


c = confusion_matrix(ytest,yp)


# In[33]:


sns.heatmap(c)


# **SVC

# In[35]:


sv = SVC()
sv.fit(xtrain,ytrain)


# In[36]:


sv.score(xtest,ytest)


# **Random Forest Classifier*

# In[37]:


rfc = RandomForestClassifier(n_estimators=200)


# In[38]:


rfc.fit(xtrain,ytrain)


# In[39]:


rfc.score(xtest,ytest)


# In[40]:


from sklearn.model_selection import cross_val_score
rfm = cross_val_score(rfc,X=xtrain,y=ytrain,cv=10)


# In[41]:


rfm.mean()


# **K nearest Neighbour**

# In[42]:


from sklearn.neighbors import KNeighborsClassifier


# In[43]:


kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(xtrain,ytrain)


# In[44]:


kn.score(xtest,ytest)


# In[45]:


df.head(1)


# In[46]:


a = [[29,1,0,120,190,0,1,130,1,1.3,0,0,0]]
kn.predict(a)


# In[ ]:




