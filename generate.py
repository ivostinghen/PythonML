import numpy as np
import pandas as pd
import time,os
import socket
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sb
# %matplotlib inline
#



##############   SAVE MODEL TO DISK ##########
df = pd.read_csv(r".\gestures\gesture.csv")
X = np.array(df.drop('target',1))
Y = np.array(df.target)

# classifier = svm.SVC(kernel = 'linear' , C=2**5)
# classifier = svm.SVC(kernel = 'linear' , C=2**5)
# classifier = svm.SVC(kernel = 'linear' ,decision_function_shape = 'cvr')
# classifier = svm.SVC()
# classifier = KNeighborsClassifier(5)
# classifier = tree.DecisionTreeClassifier()

classifier = svm.SVC(kernel='poly',degree=3)

classifier.fit(X,Y)
filename = 'modelUnity.sav'
pickle.dump(classifier, open(filename, 'wb'))
print("model saved!")


