# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 12:13:55 2016

@author: Rakshu
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from sklearn.metrics import roc_curve, auc

def encode_target(df, target_column):
    
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    mapping = {name: n for n, name in enumerate(targets)}
    df_mod[target_column].replace(mapping, inplace = True)
    
    return (df_mod, mapping)


df = pd.read_csv('E:/academics/sem 7/SVM-transport/data/data_trans_rak.csv')


# ThroughVeh_mapping = {"2W" : 0,"2w": 0 ,"AUTO":2, "JEEP":4,"AR":4,"CAR":4,"BUS":6,"VAN":4,"MINI TRUCK (ACE)":6,"TRUCK":6,"TEMPO":6}
ThroughVeh_mapping = {"2W" : "2W","2w": "2W" ,"AUTO":"AUTO", "JEEP":"CAR","AR":"AUTO","CAR":"CAR","BUS":"TRUCK","VAN":"CAR","MINI TRUCK (ACE)":"TRUCK","TRUCK":"TRUCK","TEMPO":"TRUCK"}
df['Through Veh.'].replace(ThroughVeh_mapping, inplace = True)
# Create new feature
df['Subj-Through'] = df['Sub Vehicle'].map(str) + " " + df['Through Veh.'].map(str) #Only truck auto missing

df, AccptReg_mapping = encode_target(df, 'Accpt/Reg')
df, SubjThrough_mapping = encode_target(df, 'Subj-Through')
#df, Subj_mapping = encode_target(df, 'Sub Vehicle')
#df, Through_mapping = encode_target(df, 'Through Veh.')

y = df['Accpt/Reg']
X = df[['Spatial Gap','Speed (km/hr)','Subj-Through']]
#
X_train = X[:987]
X_test = X[987:]
y_train = y[:987]
y_test = y[987:]


print "\n Fitting the classifier to the training set"
clf = RandomForestClassifier(random_state=39)
clf = clf.fit(X_train, y_train)

print "\n Predicting the people names on the testing set"
y_pred = clf.predict(X_test)
y_test_list = y_test.tolist()
y_pred_list = y_pred.tolist()

print confusion_matrix(y_test_list, y_pred_list)
#print clf.score(X_test, y_test)


print 'Performing Cross Validation.....'
cv = cross_validation.KFold(len(X), n_folds=10)
results = pd.DataFrame({'precision': [0],
                        'recall':[0],
                        'fscore': [0],
                        'accuracy':[0]})
for traincv, testcv in cv:
#    classifier = SVC(kernel='rbf', C=500, gamma=0.02).fit(X[traincv[0]:traincv[len(traincv)-1]], 
#                     y[traincv[0]:traincv[len(traincv)-1]])
    classifier = RandomForestClassifier(random_state=39).fit(X[traincv[0]:traincv[len(traincv)-1]], 
                     y[traincv[0]:traincv[len(traincv)-1]])
    pred = classifier.predict(X[testcv[0]:testcv[len(testcv)-1]])
    preds1 = classifier.predict_proba(X_test)[:,1]
    fpr1, tpr1, _ = roc_curve(y_test, preds1)
    auc1 = auc(fpr1, tpr1)
    precision, recall, fscore, support = score(y[testcv[0]:testcv[len(testcv)-1]],pred)
    accuracy = accuracy_score(y[testcv[0]:testcv[len(testcv)-1]], pred)
    print confusion_matrix(y[testcv[0]:testcv[len(testcv)-1]], pred)
    dummy = pd.DataFrame({'precision': [precision[1]],
                        'recall':[recall[1]],
                        'fscore': [fscore[1]],
                        'accuracy':[accuracy],
                        'auc':[auc1]})
    results = results.append(dummy)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    
    
    
print "Average Accuracy of CV: ", np.mean(results[1:])