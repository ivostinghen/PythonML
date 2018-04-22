import numpy as np
import pandas as pd
import time,os
import socket
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
# from sklearn import svm
# import matplotlib.pyplot as plt
# import seaborn as sb
# %matplotlib inline
#
# fileUnity = r".\communication\unity.txt";
# filePython = r".\communication\python.txt";

 
host,port = "127.0.0.1" , 10000
data = "OPEN"
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


resultadoAnterior = "";

lastTimeUnity  = "";


df = pd.read_csv(r".\gestures\gesture.csv")
X = np.array(df.drop('target',1))
Y = np.array(df.target)
# knn = svm.SVC(kernel = 'linear' , C=2**5)

# knn = svm.SVC(kernel = 'linear' , C=2**5)
# knn = svm.SVC(kernel = 'linear' ,decision_function_shape = 'cvr')

# knn = svm.SVC()
# knn = KNeighborsClassifier(5)
knn = tree.DecisionTreeClassifier()
knn.fit(X,Y)



try:

    sock.connect((host,port))
   
    while (1):
       
        #recebe input
        inputReceived = sock.recv(1024).decode("utf-8")
        line = inputReceived
        line = line.rstrip('\n')
        mylist = [float(x) for x in line.split(',')]

        #envia resultado
        result = knn.predict([mylist])
        result = result[0]
        sock.sendall(result.encode("utf-8"))
        
        # if(resultado != None):
        #     if(resultadoAnterior != resultado ):
        #         resultadoAnterior = resultado
        #         try:
        #             print(resultado)
        #             sock.sendall(data.encode("utf-8"))
        #         except Exception:
        #             print("Erro ao enviar")

        
        # print(result)
        time.sleep(.02)





finally:
    sock.close()



#
# input("Press Enter to continue...")
# predict();
