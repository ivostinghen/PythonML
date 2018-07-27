import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
import pickle
training_percentage = 0.3 # float entre 0 e 1
bestResult= [0,0,0,0,0,0,0,0,0,0]
winner = 0 
# pegar input
df = pd.read_csv(r"gestures\gesture.csv")

# pegar tamanho de t e f
# T se refere a "Teste"
# F se refere a "Fim" (por é o fim)
t_rows = int(round(df.shape[0]*training_percentage))
f_rows = df.shape[0] - t_rows

# Randomizar entrada
D = df.sample(frac=1,random_state=2).reset_index(drop=True);

# Pegar primeiros para treinamento
DT = D.iloc[:t_rows]
XT = np.array(DT.drop('target',1))
YT = np.array(DT.target)
# XT.tofile("TESTE", format="%s")
# __________________DECISION TREE_______________________

# Fazer treinamento de acordo com algoritmo
classifiers = {} 
classifiers[0] = tree.DecisionTreeClassifier()
# classifiers[1] =  svm.SVC(kernel="linear", C=0.025)
# classifiers[2] =   svm.SVC(gamma=2, C=1)
# classifiers[3] =   svm.SVC(kernel="poly", C=0.025)
# classifiers[4] =svm.SVC(kernel="sigmoid", gamma=2)
classifiers[1] = KNeighborsClassifier(3)
classifiers[2] = KNeighborsClassifier(5)
# classifiers[0] = tree.DecisionTreeClassifier(min_samples_split=1)




classifiers[3] = tree.DecisionTreeClassifier(min_samples_split=2,splitter="random",min_impurity_decrease=.000114)
classifiers[4] = tree.DecisionTreeClassifier(min_samples_split=3,splitter="random",min_impurity_decrease=.000116)
classifiers[5] = tree.DecisionTreeClassifier(min_samples_split=4,splitter="random",min_impurity_decrease=.000118)
classifiers[6] = tree.DecisionTreeClassifier(min_samples_split=5,splitter="random",min_impurity_decrease=.00012)
classifiers[7] = tree.DecisionTreeClassifier(min_samples_split=6,splitter="random",min_impurity_decrease=.000122)
classifiers[8] = tree.DecisionTreeClassifier(min_samples_split=2,splitter="random",min_impurity_decrease=.000114)

classifiers[9] = tree.DecisionTreeClassifier(min_samples_split=3,splitter="best",min_impurity_decrease=.000114)
classifiers[10] = tree.DecisionTreeClassifier(min_samples_split=4,splitter="best",min_impurity_decrease=.000116)
classifiers[11] = tree.DecisionTreeClassifier(min_samples_split=5,splitter="best",min_impurity_decrease=.00012)
classifiers[12] = tree.DecisionTreeClassifier(min_samples_split=6,splitter="best",min_impurity_decrease=.000118)

classifiers[13] = tree.DecisionTreeClassifier(min_samples_split=2,splitter="best")
classifiers[14] = tree.DecisionTreeClassifier(min_samples_split=3,splitter="best")
classifiers[15] = tree.DecisionTreeClassifier(min_samples_split=4,splitter="best")
classifiers[16] = tree.DecisionTreeClassifier(min_samples_split=5,splitter="best")
classifiers[17] = svm.SVC(kernel='poly' , degree=3)#default
# classifiers[10] = pickle.load(open('10.sav', 'rb'))
# classifiers[0] = pickle.load(open('0.sav', 'rb'))
# classifiers[1] = pickle.load(open('1.sav', 'rb'))
# classifiers[2] = pickle.load(open('2.sav', 'rb'))
# classifiers[3] = pickle.load(open('3.sav', 'rb'))
# classifiers[4] = pickle.load(open('4.sav', 'rb'))
# classifiers[5] = pickle.load(open('5.sav', 'rb'))
# classifiers[6] = pickle.load(open('6.sav', 'rb'))
# classifiers[7] = pickle.load(open('7.sav', 'rb'))
# classifiers[8] = pickle.load(open('8.sav', 'rb'))
# classifiers[9] = pickle.load(open('9.sav', 'rb'))
# classifiers[10] = pickle.load(open('10.sav', 'rb'))

# classifiers[0] = tree.DecisionTreeClassifier(min_samples_split=2,splitter="random",min_impurity_decrease=.000114)
# classifiers[1] = tree.DecisionTreeClassifier(min_samples_split=3,splitter="random",min_impurity_decrease=.000116)
# classifiers[2] = tree.DecisionTreeClassifier(min_samples_split=4,splitter="random",min_impurity_decrease=.000118)
# classifiers[3] = tree.DecisionTreeClassifier(min_samples_split=5,splitter="random",min_impurity_decrease=.00012)
# classifiers[4] = tree.DecisionTreeClassifier(min_samples_split=6,splitter="random",min_impurity_decrease=.000122)
# classifiers[5] = tree.DecisionTreeClassifier(min_samples_split=2,splitter="random",min_impurity_decrease=.000114)

# classifiers[6] = tree.DecisionTreeClassifier(min_samples_split=3,splitter="best",min_impurity_decrease=.000114)
# classifiers[7] = tree.DecisionTreeClassifier(min_samples_split=4,splitter="best",min_impurity_decrease=.000116)
# classifiers[8] = tree.DecisionTreeClassifier(min_samples_split=5,splitter="best",min_impurity_decrease=.00012)
# classifiers[9] = tree.DecisionTreeClassifier(min_samples_split=6,splitter="best",min_impurity_decrease=.000118)
# classifiers[10] = pickle.load(open('model.sav', 'rb'))















# classifiers[5] = tree.DecisionTreeClassifier(min_samples_split=7,splitter="random",min_impurity_decrease=.00016)


# classifiers[6] = tree.DecisionTreeClassifier(min_samples_split=8,splitter="random",min_impurity_decrease=.00017)

# classifiers[8] = tree.DecisionTreeClassifier(min_samples_split=10,splitter="random",min_impurity_decrease=.09)

# classifiers[2] = tree.DecisionTreeClassifier(0.8)

# # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
# classifiers[3] = svm.SVC(kernel="rbf")#default
# classifiers[4] =  svm.SVC(kernel="linear")
# classifiers[5] = svm.SVC(kernel='poly' , degree=2)
# classifiers[6] = svm.SVC(kernel='poly' , degree=3)#default
# classifiers[7] = svm.SVC(kernel='poly' , degree=5)
# classifiers[0] = svm.SVC(kernel='poly' , degree=7)
# classifiers[1] = svm.SVC(kernel='poly' , degree=9)
# classifiers[9] = svm.SVC(kernel='poly' , degree=9, gamma=0;0)
# classifiers[0] = svm.SVC(kernel='sigmoid')
# classifiers[0] = svm.SVC(kernel='precomputed')
# classifiers[4] = svm.SVC(kernel='callable')

# classifiers[2] = svm.SVC(kernel='rbf')

# # classifiers[2] = KNeighborsClassifier(3)

# classifiers[1] = svm.LinearSVC()
# classifiers[2] = svm.SVC(kernel='linear',gamma=30)
# classifiers[1] = svm.SVC(kernel='poly')
# classifiers[1] = svm.SVC(kernel='linear')
# classifiers[2] = svm.SVC(kernel='poly')
# classifiers[2] = tree.DecisionTreeClassifier()

#Treina os classificadores

classifierNames = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]
# classifierNames = ["knn(3)","knn(5)","DecisionTree", "SVC_rbf","SVC_linear" , "SVC_degree_2","SVC_degree_3","SVC_degree_5","SVC_degree_7" ,"SVC_degree_9" ]
gestures = ["OPEN","CLOSE","THUMB","TWO","THREE","FOUR","LOVE","COOL","FIRE", "ITALIAN", "HANG_LOOSE"]


for c in classifiers:

	# if(c!=10):
	classifiers[c].fit(XT,YT)
	
	# filename = classifierNames[c]+'.sav'
	# pickle.dump(classifiers[c], open(filename, 'wb'))
	# # print("model saved!")



	# Pegar últimos % para testes
	D2 = D.iloc[-f_rows:]
	XF = np.array((D2).drop('target',1))
	YF = np.array(D2[D2.columns[-1]])


	print("\nClassifier:"+classifierNames[c] )
	acertos = [0,0,0,0,0,0,0,0,0,0,0]
	erros = [0,0,0,0,0,0,0,0,0,0,0]
	total = [0,0,0,0,0,0,0,0,0,0,0]
	temp = 0 

	for i in range(0, f_rows) :
		result = classifiers[c].predict([XF[i]]);
		# print(YF[i])
		if YF[i] == "OPEN":
			temp=0
		elif YF[i] ==  "CLOSE":
			temp=1
		elif YF[i] ==  "THUMB":
			temp=2
		elif YF[i] == "TWO":
			temp=3
		elif YF[i] == "THREE":
			temp=4
		elif YF[i] ==  "FOUR":
			temp=5
		elif YF[i] ==  "LOVE":
			temp=6
		elif YF[i] ==  "COOL":
			temp=7
		elif YF[i] ==  "FIRE":
			temp=8
		elif YF[i] ==  "ITALIAN":
			temp=9
		elif YF[i] ==  "HANG_LOOSE":
			temp=10
	
		total[temp] +=1;
		if (result == YF[i]):
			acertos[temp] += 1


	resultArray = []
	for x in range(0,10): 
		result = round((100*(acertos[x]/total[x])),2)
		resultArray.append(result) 


	pointA=0
	pointB=0
	for x in range(0,10):
		if(resultArray[x]>bestResult[x]):
			pointA+=1
		else:
			pointB+=1
	if(pointA>pointB):
		bestResult = resultArray[:]
		winner = c
			



	print(resultArray)

print('\n',bestResult , '\n' ,winner)