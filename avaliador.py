import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm

training_percentage = 0.3 # float entre 0 e 1

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


# __________________DECISION TREE_______________________

# Fazer treinamento de acordo com algoritmo
classifier = tree.DecisionTreeClassifier()
classifier.fit(XT,YT)

# Pegar últimos % para testes
D2 = D.iloc[-f_rows:]
XF = np.array((D2).drop('target',1))
YF = np.array(D2[D2.columns[-1]])

acertos = 0

# Fazer testes
for i in range(0, f_rows) :
	result = classifier.predict([XF[i]]);
	if (result == YF[i]):
		acertos += 1;
# Mostrar resultados
print("DecisionTree:")
print ("total de testes = " + str(f_rows))
print ("acertos = " + str(acertos))
print ("porcentagem de acertos = " + str(100*acertos/f_rows) + "%")

# __________________KNN(5)_______________________


classifier = KNeighborsClassifier(5)
# classifier = svm.SVC()
classifier.fit(XT,YT)

acertos = 0

# Fazer testes
for i in range(0, f_rows) :
	result = classifier.predict([XF[i]]);
	if (result == YF[i]):
		acertos += 1;
# Mostrar resultados
print("KNN 5 VIZINHOS:")
print ("total de testes = " + str(f_rows))
print ("acertos = " + str(acertos))
print ("porcentagem de acertos = " + str(100*acertos/f_rows) + "%")
print("")
# __________________KNN(3)_______________________


classifier = KNeighborsClassifier(3)
# classifier = svm.SVC()
classifier.fit(XT,YT)

acertos = 0

# Fazer testes
for i in range(0, f_rows) :
	result = classifier.predict([XF[i]]);
	if (result == YF[i]):
		acertos += 1;
# Mostrar resultados
print("KNN 3 VIZINHOS:")
print ("total de testes = " + str(f_rows))
print ("acertos = " + str(acertos))
print ("porcentagem de acertos = " + str(100*acertos/f_rows) + "%")
print("")

# __________________KNN(7)_______________________


classifier = KNeighborsClassifier(7)
# classifier = svm.SVC()
classifier.fit(XT,YT)

acertos = 0

# Fazer testes
for i in range(0, f_rows) :
	result = classifier.predict([XF[i]]);
	if (result == YF[i]):
		acertos += 1;
# Mostrar resultados
print("KNN 7 VIZINHOS:")
print ("total de testes = " + str(f_rows))
print ("acertos = " + str(acertos))
print ("porcentagem de acertos = " + str(100*acertos/f_rows) + "%")
print("")

# __________________SVC_______________________


classifier = svm.SVC()
classifier.fit(XT,YT)

acertos = 0

# Fazer testes
for i in range(0, f_rows) :
	result = classifier.predict([XF[i]]);
	if (result == YF[i]):
		acertos += 1;
# Mostrar resultados
print("SVM:")
print ("total de testes = " + str(f_rows))
print ("acertos = " + str(acertos))
print ("porcentagem de acertos = " + str(100*acertos/f_rows) + "%")
print("")