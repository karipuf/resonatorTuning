#########################################
# Parsing arguments and loading modules
#########################################

from argparse import ArgumentParser
ap=ArgumentParser()
ap.add_argument('-n',help="Number of parameter sets to attempt (default 50)")
ap.add_argument('-e',help='Number of epochs to train for (default 800)')
ap.add_argument('-f',help='Number of filters in first convolution unit (default: sample from [32,64,96])')
ap.add_argument('--x1',help='Width of first convolutional filter (default: sample from [5,7,9,11])')
ap.add_argument('-x',help='Width of subsequent convolutional filters (default: sample from [3,5,7,9])')
ap.add_argument('--fc6',help='Number of hidden units in fc6 (default: sample from [64,96,128,256]')
ap.add_argument('--lr',help='Initial learning rate (default: sample from [0.001,0.002,0.0005,0.0001])')
ap.add_argument('--l2',help='L2 Regularization constant (default 0.00001)')
ap.add_argument('-d',help='Number of convolution layers (default: sample from [2,3,4,5])')
ap.add_argument('--rs',help='Random state for param sampler (default=10)')
ap.add_argument('-o',help='Name of file to print result')
parsed=ap.parse_args()

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

def CreatePredictor(paramVec,l2reg,dropProb=.5):

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

###################################
# Training, parameter search, etc
###################################

# Options
if parsed.o==None: outFile='results.txt'
else: outFile=parsed.o
nParams=ProcessArg(parsed.n,50)

pDict={'f':ProcessArg(parsed.f,[32,64,96]),
       'x1':ProcessArg(parsed.x1,[5,7,9,11]),
       'x':ProcessArg(parsed.x,[3,5,7,9]),
       'fc6':ProcessArg(parsed.fc6,[64,96,128,256]),
       'lr':ProcessArg(parsed.lr,[0.001,0.002,0.0005,0.0001]),
       'd':ProcessArg(parsed.d,[2,3,4,5])}
pSamper=iter(ParameterSampler(pDict,n_iter=nParams,random_state=ProcessArg(parsed.rs,10)))

# Loading in the data
resp=pd.read_csv("Data.txt",header=None,sep='\s+').T
resp.columns=resp.iloc[0,:]
resp=resp.iloc[1:,:]
resp=(resp-resp.mean())/resp.std()
curr=pd.read_csv("Parameters.txt",header=None,sep='\s+')

# Removing ideal resp and current
idealResp=resp.iloc[0,:]
resp=resp.iloc[1:,:]
idealCurr=curr.iloc[0,:]
curr=curr.iloc[1:,:]

# Creating test and training sets
xtrain,xtest,ytrain,ytest=train_test_split(resp,curr,test_size=.05)

# Creating and training models!
results=[]
for count in range(nParams):
    #pdb.set_trace()
    pVec=next(pSamper)
    mod=CreatePredictor(pVec,ProcessArg(parsed.l2,0.00001))
    mod.fit(xtrain.values.reshape((-1,256,1)),ytrain.values,batch_size=128,epochs=ProcessArg(parsed.e,800))
    ofile=open(outFile,'a+')
    ofile.write(str(pVec)+'\n')
    ofile.write('Score: '+str(mod.evaluate(xtest.values.reshape((-1,256,1)),ytest))+'\n')
    ofile.close()
    
