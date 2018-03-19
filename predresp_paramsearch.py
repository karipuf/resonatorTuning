#########################################
# Parsing arguments and loading modules
#########################################

from argparse import ArgumentParser
ap=ArgumentParser()
ap.add_argument('-n',help="Number of parameter sets to attempt (default 50)")
ap.add_argument('-e',help='Number of epochs to train for (default 1200)')
ap.add_argument('-f',help='Number of filters in first convolution unit (default: sample from [32,64,96])')
ap.add_argument('--x1',help='Width of first convolutional filter (default: sample from [5,7,9,11])')
ap.add_argument('-x',help='Width of subsequent convolutional filters (default: sample from [3,5,7,9])')
ap.add_argument('--fc6',help='Number of hidden units in fc6 (default: sample from [64,96,128,256]')
ap.add_argument('--lr',help='Initial learning rate (default: sample from [0.001,0.002,0.0005,0.0001])')
ap.add_argument('--l2',help='L2 Regularization constant (default 0.00001)')
ap.add_argument('-d',help='Number of convolution layers (default: sample from [2,3,4,5])')
ap.add_argument('--rs',help='Random state for param sampler (default=10)')
ap.add_argument('--drop',help='dropout probability (default [.2,.3,.4,.5,.6]])')
ap.add_argument('-o',help='Name of file to print result')
parsed=ap.parse_args()

from predresp_funcs import *

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
       'd':ProcessArg(parsed.d,[2,3,4,5]),
       'drop':ProcessArg(parsed.drop,[.2,.3,.4,.5,.6]),
       'l2':ProcessArg(parsed.l2,[.00001,.00005,.00005])
}

# Creating and training models!
pSamper=iter(ParameterSampler(pDict,n_iter=nParams,random_state=ProcessArg(parsed.rs,10)))
for count in range(nParams):
    
    pVec=next(pSamper)
    
    try:

        tmpScore=TestPredictor(pVec,nEpochs=ProcessArg(parsed.e,1200))
        ofile=open(outFile,'a+')
        ofile.write(str(pVec)+'\n')
        ofile.write('Score: '+str(tmpScore)+'\n')
        ofile.close()
        
    except ValueError:

        ef=open("ErrorFile","a+")
        ef.write('Error encountered! pVec is: '+str(pVec)+'\n')
        ef.close()
        
