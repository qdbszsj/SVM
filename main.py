

import pandas as pd
import numpy as np
dataset = pd.read_csv('/home/parker/watermelonData/watermelon3_0a.csv', delimiter=",")
# print(dataset)

X=dataset.iloc[range(17),[1,2]].values
y=dataset.values[:,3]
print("trueV",y)
trueV=y
from sklearn import svm

linearKernalMethod=svm.SVC(C=10000,kernel='linear')#C=1 defaultly
linearKernalMethod.fit(X,y)
predictV=linearKernalMethod.predict(X)

print("linear",predictV)

confusionMatrix=np.zeros((2,2))
for i in range(len(y)):
    if predictV[i]==trueV[i]:
        if trueV[i]==0:confusionMatrix[0,0]+=1
        else: confusionMatrix[1,1]+=1
    else:
        if trueV[i]==0:confusionMatrix[0,1]+=1
        else:confusionMatrix[1,0]+=1
print("linearConfusionMatrix\n",confusionMatrix)


rbfKernalMethod=svm.SVC(C=10000,kernel='rbf')#C=1 defaultly
rbfKernalMethod.fit(X,y)
predictV=rbfKernalMethod.predict(X)

print("rbf",predictV)

confusionMatrix=np.zeros((2,2))
for i in range(len(y)):
    if predictV[i]==trueV[i]:
        if trueV[i]==0:confusionMatrix[0,0]+=1
        else: confusionMatrix[1,1]+=1
    else:
        if trueV[i]==0:confusionMatrix[0,1]+=1
        else:confusionMatrix[1,0]+=1
print("rbfConfusionMatrix\n",confusionMatrix)

