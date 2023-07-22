#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df_train=pd.read_csv('A:\\train.csv')


# In[3]:


df_test=pd.read_csv('A:\\test.csv')


# In[4]:


df_train.drop('ID',inplace=True,axis=1)


# In[5]:


df_train.columns


# In[6]:


df_train.head()


# In[7]:


df_test.drop('ID',inplace=True,axis=1)


# In[8]:


df_test.head()


# # 1.variance ==0 remove them
# 

# In[9]:


zero_var=df_train.var()[df_train.var()==0].index.values


# In[10]:


df_train.drop(zero_var,inplace=True,axis=1)
df_test.drop(zero_var,inplace=True,axis=1)


# # 2. check for null and unique
# 

# In[11]:


np.sum(df_train.isnull().sum())


# In[12]:


np.sum(df_test.isnull().sum())


# In[13]:


#check for unique


# In[14]:


for i in df_train.columns:
    print(df_train[i].unique())


# In[15]:


label_columns=df_train.describe(include='object').columns.values


# # 3.Apply label encoder on categorical columns
# 

# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


le=LabelEncoder()


# In[18]:


for col in label_columns:
    le.fit(df_train[col].append(df_test[col]))
    df_train[col]=le.transform(df_train[col])
    df_test[col]=le.transform(df_test[col])


# In[19]:


df_train['X0'].unique()


# # 4. performing dimensionality reduction

# In[20]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# In[21]:


pca=PCA(n_components=0.98)


# In[22]:


x=df_train.drop('y',axis=1)
y=df_train['y']


# In[23]:


y


# In[24]:


X_train,X_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)


# In[25]:


pca.fit(x)


# In[26]:


pca.n_components_


# In[27]:


pca.explained_variance_ratio_


# # 5. predicting using XG boost regressor

# In[28]:


pca_xtrain=pca.transform(X_train)
pca_xval=pca.transform(X_val)


# In[29]:


from xgboost import XGBRegressor


# In[30]:


XG=XGBRegressor()


# In[31]:


XG.fit(pca_xtrain,y_train)


# In[32]:


predict=XG.predict(pca_xval)


# In[34]:


predict


# In[37]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


# In[38]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[41]:


scores = cross_val_score(XG, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)


# In[43]:


from numpy import absolute


# In[44]:


scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


# In[ ]:




