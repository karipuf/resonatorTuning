from argparse import ArgumentParser
ap=ArgumentParser()
ap.add_argument('-n',help='Number of function calls (default=35)')
ap.add_argument('-e',help='Number of epochs (default=1200)')
ap.add_argument('-o',help='Output file for results (default skopt_result.txt')
parsed=ap.parse_args()

from predresp_funcs import *
from skopt import gp_minimize
from skopt.space import Real,Integer
import pickle

# Params
nCalls=ProcessArg(parsed.n,35)
nEpochs=ProcessArg(parsed.e,1200)
if parsed.o==None:
    resFile='skopt_result.txt'
else:
    resFile=parsed.o

# Setting up
counter=iter(range(1000))
tf=open("skopt_tmp.txt","a+")
def sp(tmp):
    tf.write("Scikit-opt Round #"+str(next(counter))+"\n")
    tf.flush()
    return SP(tmp,nEpochs=nEpochs)

# Optimizing!!
space=[Integer(32,256),Integer(2,4),Real(0.00001,0.01),Integer(32,512),Integer(3,12),Integer(3,12),Real(.2,.6),Real(0.000001,0.001),Integer(1,3)]
res=gp_minimize(sp,space,n_calls=nCalls)

# Wrapping up
print(str(res))
pickle.dump([res['x'],res['fun']],open(resFile,"wb+"))

