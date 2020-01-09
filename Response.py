#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[ ]:





# #  1. Business understanding

# In[64]:


cloth_data=pd.read_csv('D:\Asg\Clothing_Store.csv')
cloth_data.head(10)


# In[65]:


print  (str(len(cloth_data.index)))
#total number of records


# # 2. Data understanding

# In[66]:


cloth_data.info()
sns.countplot(x="RESP", data=cloth_data)
#dependent variable


# In[67]:


sns.countplot(x="RESP", hue="MAILED", data=cloth_data)


# In[68]:


sns.countplot(x="RESP", hue="RESPONDED", data=cloth_data)


# In[69]:


cloth_data["PROMOS"].plot.hist()


# In[70]:


cloth_data["RESPONSERATE"].plot.hist()


# # Data preparation

# In[71]:


cloth_data.isnull()


# In[72]:


cloth_data.isnull().sum()
# No null value is found


# In[73]:


sns.boxplot(x='RESP', y='RESPONSERATE', data=cloth_data)


# In[74]:


cloth_data.head(5)


# In[75]:


#cloth_data.drop('HHKEY', axis=1, inplace=True)
#cloth_data.drop('ZIP_CODE', axis=1, inplace=True)
cloth_data.head(5)


# In[76]:


cloth_data.drop(['REC','CC_CARD'], axis=1, inplace=True)


# In[77]:


cloth_data.drop(['PBLOUSES','VALPHON','WEB','HI','LTFREDAY','CLUSTYPE','PERCRET','PC_CALC20','PSWEATERS','PKNIT_TOPS', 'PKNIT_DRES'], axis=1, inplace=True)


# In[78]:


cloth_data.head(5)


# In[79]:


cloth_data.drop(['PJACKETS','PCAR_PNTS','PCAS_PNTS','PSHIRTS','PDRESSES','PSUITS','POUTERWEAR','MARKDOWN','CLASSES','COUPONS','STYLES','STORES','STORELOY'], axis=1,inplace=True)


# In[80]:


cloth_data.head(5)


# In[81]:


cloth_data.drop(['PJEWELRY','PFASHION','PLEGWEAR','PCOLLSPND','AMSPEND','PSSPEND','CCSPEND','SMONSPEND','PREVPD','GMP'], axis=1, inplace=True)


# In[82]:


cloth_data.head(5)


# In[83]:


cloth_data.drop(['AXSPEND','TMONSPEND','OMONSPEND','DAYS','FREDAYS'], axis=1, inplace=True)


# In[91]:


cloth_data.head(5)


# In[92]:


cloth_data.drop(['HHKEY','ZIP_CODE'],axis=1, inplace=True)


# In[95]:


cloth_data.head(5)


# # Modelling

# In[96]:


#Training

X=cloth_data.drop("RESP", axis=1)
y = cloth_data["RESP"]



# In[104]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[130]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[131]:


from sklearn.linear_model import LogisticRegression


# In[107]:


logmodel=LogisticRegression() 


# In[108]:


logmodel.fit(X_train, y_train)


# In[109]:


model = LogisticRegression(solver='liblinear')
model = LogisticRegression(solver='lbfgs')


# In[119]:


predictions=logmodel.predict(X_train)


# In[120]:


from sklearn.metrics import classification_report


# In[ ]:





# In[129]:


classification_report(y_test, predictions)


# In[1]:


from sklearn.metrics import confusion_matrix


# In[2]:


confusion_matrix(y_test,predictions)


# In[3]:


from sklearn.metrics import accuracy_score


# In[4]:


accuracy_score(y_test,predictions)


# In[ ]:




