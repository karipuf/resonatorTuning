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

def GetData(testSize=.05,randomState=10,normalize=True):

    # Loading in the data (Response)
    resp=pd.read_csv("Data.txt",header=None,sep='\s+').T
    resp.columns=resp.iloc[0,:]
    resp=resp.iloc[1:,:]
    if normalize:
        resp=(resp-resp.mean())/resp.std()

    # "Current" (actually it's the phase and coupling ratios)
    curr=pd.read_csv("Parameters.txt",header=None,sep='\s+')
    
    # Removing ideal resp and current
    idealResp=resp.iloc[0,:]
    resp=resp.iloc[1:,:]
    idealCurr=curr.iloc[0,:]
    curr=curr.iloc[1:,:]

    # Creating test and training sets
    xtrain,xtest,ytrain,ytest=train_test_split(resp,curr,test_size=testSize,random_state=randomState)
    return xtrain,xtest,idealResp,ytrain,ytest,idealCurr
    
def CreatePredictor(paramVec): #,l2reg=.00001):

    nFilt=paramVec['f']
    nLayers=paramVec['d']
    learnRate=paramVec['lr']
    fc6hidden=paramVec['fc6']
    filtWidth1=paramVec['x1']
    filtWidth=paramVec['x']
    dropProb=paramVec['drop']
    l2reg=paramVec['l2']
    filtAdd=paramVec['fa']
    
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
        mod.add(Conv1D((filtAdd+count)*nFilt,(filtWidth,),activation='relu',kernel_regularizer=l2))
        mod.add(AveragePooling1D(2))
        mod.add(layers.Dropout(dropProb))

    mod.add(layers.Flatten())
    mod.add(Dense(fc6hidden,activation='relu'))
    mod.add(layers.Dropout(dropProb))
    mod.add(Dense(8,activation='linear'))
    
    mod.compile(optimizer=Adam(lr=learnRate),loss='mse')
    return mod

def TestPredictor(paramVec,randomState=10,nEpochs=1200,batchSize=128,modelFile=None,data=None,valSplit=.05,callbacks=[]):

    if data==None:
        xtrain,xtest,idealResp,ytrain,ytest,idealCurr=GetData(randomState=randomState)
        mod=CreatePredictor(paramVec)
    else:
        (xtrain,xtest,ytrain,ytest)=data;

    mod.fit(xtrain.values.reshape((-1,256,1)),ytrain.values,batch_size=128,epochs=nEpochs,validation_split=valSplit,callbacks=callbacks)

    if modelFile!=None:
        mod.save(modelFile)
    
    return (mod,mod.evaluate(xtest.values.reshape((-1,256,1)),ytest.values))


def ParamList2Dict(paramList):
    
    (f,d,lr,fc6,x1,x,drop,l2,fa)=paramList
    paramVec={}
    paramVec['f']=f
    paramVec['d']=d
    paramVec['lr']=lr
    paramVec['fc6']=fc6
    paramVec['x1']=x1
    paramVec['x']=x
    paramVec['drop']=drop
    paramVec['l2']=l2
    paramVec['fa']=fa

    return paramVec

def ParamDict2List(paramVec):

    f=paramVec['f']
    d=paramVec['d']
    lr=paramVec['lr']
    fc=paramVec['fc6']
    x1=paramVec['x1']
    x=paramVec['x']
    drop=paramVec['drop']
    l2=paramVec['l2']
    fa=paramVec['fa']

    return (f,d,lr,fc6,x1,x,drop,l2,fa)

# Simple version to use with scikit-opt
def SP(paramList,nEpochs=1200,valSplit=.05,callbacks=[]):

    #(f,d,lr,fc6,x1,x,drop,l2,fa)=paramList
    #paramVec={}
    #paramVec['f']=f
    #paramVec['d']=d
    #paramVec['lr']=lr
    #paramVec['fc6']=fc6
    #paramVec['x1']=x1
    #paramVec['x']=x
    #paramVec['drop']=drop
    #paramVec['l2']=l2
    #paramVec['fa']=fa

    paramVec=ParamList2Dict(paramList)
    return TestPredictor(paramVec,nEpochs=nEpochs,valSplit=valSplit,callbacks=callbacks)[1]
    
