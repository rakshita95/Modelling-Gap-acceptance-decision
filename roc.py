# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 00:40:08 2016

@author: prasanna_pothuganti
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np

def encode_target(df, target_column):
    
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    mapping = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(mapping, inplace = True)
    
    return (df_mod, mapping)

df = pd.read_csv('E:/academics/sem 7/SVM-transport/data/data_trans_rak.csv')

ThroughVeh_mapping = {"2W" : "2W","2w": "2W" ,"AUTO":"AUTO", "JEEP":"CAR","AR":"AUTO","CAR":"CAR","BUS":"TRUCK","VAN":"CAR","MINI TRUCK (ACE)":"TRUCK","TRUCK":"TRUCK","TEMPO":"TRUCK"}
df['Through Veh.'].replace(ThroughVeh_mapping, inplace = True)
# Create new feature
df['Subj-Through'] = df['Sub Vehicle'].map(str) + " " + df['Through Veh.'].map(str) #Only truck auto missing
df, AccptReg_mapping = encode_target(df, 'Accpt/Reg')
df, SubjThrough_mapping = encode_target(df, 'Subj-Through')

y = df['Accpt/Reg']
X = df[['Spatial Gap','Speed (km/hr)','Subj-Through']]
#
X_train = X[:987]
X_test = X[987:]
y_train = y[:987]
y_test = y[987:]


print "\n Fitting the classifier to the training set"
clf1 = SVC(kernel='rbf', C=500, gamma=0.02, probability = True)
clf1 = clf1.fit(X_train, y_train)
print "\n Predicting the people names on the testing set"
y_pred1 = clf1.predict(X_test)
print confusion_matrix(y_test, y_pred1)

from sklearn.metrics import roc_curve, auc

preds1 = clf1.predict_proba(X_test)[:,1]
fpr1, tpr1, _ = roc_curve(y_test, preds1)
auc1 = auc(fpr1, tpr1)

print "\n Fitting the classifier to the training set"
clf2 = RandomForestClassifier(random_state=39)
clf2 = clf2.fit(X_train, y_train)
print "\n Predicting the people names on the testing set"
y_pred2 = clf2.predict(X_test)
print confusion_matrix(y_test, y_pred2)


preds2 = clf2.predict_proba(X_test)[:,1]
fpr2, tpr2, _ = roc_curve(y_test, preds2)
auc2 = auc(fpr2, tpr2)

