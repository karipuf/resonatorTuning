
# coding: utf-8

# In[13]:

import keras
from keras import models,layers,optimizers
from keras.layers import Conv1D,Dense,MaxPool1D,AveragePooling1D,GlobalAveragePooling1D,BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import pylab as pl
import numpy as np

#get_ipython().magic('matplotlib notebook')


# In[2]:


resp=pd.read_csv("Data.txt",header=None,sep='\s+').T
resp.columns=resp.iloc[0,:]
resp=resp.iloc[1:,:]
resp=(resp-resp.mean())/resp.std()

curr=pd.read_csv("Parameters.txt",header=None,sep='\s+')
#curr=(curr-curr.mean())/curr.std()


# In[3]:


xtrain,xtest,ytrain,ytest=train_test_split(resp,curr,test_size=.05)


# In[25]:


l2=keras.regularizers.l2(.00001)
mod=models.Sequential()
mod.add(Conv1D(2*64,(7,),activation='relu',input_shape=(256,1),kernel_regularizer=l2))
mod.add(AveragePooling1D(2))
#mod.add(BatchNormalization())
mod.add(layers.Dropout(.5))
mod.add(Conv1D(2*128,(7,),activation='relu',kernel_regularizer=l2))
mod.add(AveragePooling1D(2))
#mod.add(BatchNormalization())
mod.add(layers.Dropout(.5))
mod.add(Conv1D(2*192,(5,),activation='relu',kernel_regularizer=l2))
#mod.add(GlobalAveragePooling1D())
mod.add(layers.Flatten())
mod.add(Dense(128,activation='relu'))
mod.add(layers.Dropout(.5))
mod.add(Dense(8,activation='linear'))


# In[26]:


mod.compile(optimizer=Adam(lr=0.0005),loss='mse')


# In[27]:


mod.fit(resp.values.reshape((-1,256,1)),curr.values,batch_size=256,epochs=1000,validation_split=.1)


# In[79]:


#get_ipython().magic('pinfo layers.Dropout')

