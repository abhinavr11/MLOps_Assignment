import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json


df = pd.read_csv("../data/creditcard.csv")


dfxtrain , dfxtest = train_test_split( df, train_size=0.8, random_state=10)

dfxtrain.to_csv("../data/prepared/train.csv") #restore path
dfxtest.to_csv("../data/prepared/test.csv")   #restore path

xtrain = dfxtrain.drop(['Time','Class'], axis=1)
xtest = dfxtest.drop(['Time','Class'], axis=1)

ytrain = dfxtrain['Class']
ytest = dfxtest['Class']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,criterion='entropy')

clf.fit(xtrain, ytrain)

Pkl_Filename = "../models/model.pkl"  #edit patgh

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

acc = clf.score(xtest, ytest)

y_pred = clf.predict(xtest)

f1score = f1_score(ytest, y_pred, average='weighted')


dictionary ={
"Accuracy" : acc,
"f1 Score" : f1score,

}

with open("../metrics/acc_f1.json", "w") as outfile:
    json.dump(dictionary, outfile)
