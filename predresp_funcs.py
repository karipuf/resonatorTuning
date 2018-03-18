import keras,pdb
from keras import models,layers,optimizers
from keras.layers import Conv1D,Dense,MaxPool1D,AveragePooling1D,GlobalAveragePooling1D,BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split,ParameterSampler
import pandas as pd
import pylab as pl
import numpy as np

#############
# Functions
#############

def ProcessArg(tmp,defaultVal):
    if tmp==None: return defaultVal
    else: return eval(tmp)

def CreatePredictor(paramVec,l2reg=.00001,dropProb=.5):

    nFilt=paramVec['f']
    nLayers=paramVec['d']
    learnRate=paramVec['lr']
    fc6hidden=paramVec['fc6']
    filtWidth1=paramVec['x1']
    filtWidth=paramVec['x']

    #keepProb=paramVec['dp']
    #l2reg=paramVec['l2']

    l2=keras.regularizers.l2(l2reg)
    mod=models.Sequential()

    # Adding the first layer
    mod.add(Conv1D(nFilt,(filtWidth1,),activation='relu',input_shape=(256,1),kernel_regularizer=l2))
    mod.add(AveragePooling1D(2))
    mod.add(layers.Dropout(dropProb))

    # Adding subsequent layers
    for count in range(1,nLayers):
        mod.add(Conv1D(count*nFilt,(filtWidth,),activation='relu',kernel_regularizer=l2))
        mod.add(AveragePooling1D(2))
        mod.add(layers.Dropout(dropProb))

    mod.add(layers.Flatten())
    mod.add(Dense(fc6hidden,activation='relu'))
    mod.add(layers.Dropout(dropProb))
    mod.add(Dense(8,activation='linear'))
    
    mod.compile(optimizer=Adam(lr=learnRate),loss='mse')
    return mod
