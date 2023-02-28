#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("Bike_Data.csv")


# In[2]:


data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[6]:


X = data.drop(["price"],axis=1)
y = data["price"]


# In[7]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)


# In[8]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)


# In[9]:


r2 = r2_score(y_test,y_predict)
mae = mean_squared_error(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)


# In[10]:


import joblib
import tarfile


# In[11]:


joblib.dump(model, 'model.joblib')


# In[12]:


with tarfile.open('model.tar.gz', 'w:gz') as tar:
    tar.add('model.joblib', arcname='model.joblib')


# In[13]:


tar.close()


# In[14]:


with tarfile.open('model.tar.gz', 'r:gz') as tar:
    tar.extract('model.joblib')


# In[15]:


model = joblib.load('model.joblib')


# In[16]:


predictions = model.predict(X_test)


# In[17]:


predictions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




