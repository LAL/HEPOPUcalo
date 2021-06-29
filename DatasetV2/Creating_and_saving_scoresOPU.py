#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from permutationimportancephysics.PermutationImportance import PermulationImportance

plt.close()

# In[ ]:


# Type considered, should be ttbar or W
process_type = 'ttbar'
# Repository containing the outputs of the Splitter
exp_dir = '/data/atlas/struillou/StageM1VictorStruillou/SortieSplitter_ttbar/'
# Repository containing the outputs of the Splitter 2 (if they exist)
exp_dir_2= '/data/atlas/struillou/StageM1VictorStruillou/SortieSplitter_2_ttbar/'
# Number of events and of random projections to be used for the ROC curves. They must be smaller or equal to the number
# of events and random features produced by opu_output_analysis.
nevents = 100000
ncomp = 30000
train_size=0.8


# In[ ]:


nevents_train=80000 #must be coherent with the train_size
nevents_test=20000


# In[ ]:


with np.load(exp_dir+'traintest.npz') as data:
    X_train = data['X_train'][:nevents_train, :ncomp]
    X_test = data['X_test'][:nevents_test, :ncomp]
    y_train = data['y_train'][:nevents_train]
    y_test = data['y_test'][:nevents_test]
    weights_train = data['weights_train'][:nevents_train]
    weights_test = data['weights_test'][:nevents_test]
    evid_train = data['evid_train'][:nevents_train]
    evid_test = data['evid_test'][:nevents_test]
    

# In[ ]:

class_weights_train = (weights_train[y_train == 0].sum(), weights_train[y_train == 1].sum())

for i in range(len(class_weights_train)):
    #training dataset: equalize number of background and signal
    weights_train[y_train == i] *= max(class_weights_train)/ class_weights_train[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test[y_test == i] *= 1/(1-train_size)

clh = SGDClassifier(loss="log",fit_intercept=True)
gsa = GridSearchCV(clh, param_grid={"alpha": np.linspace(310,370,num=10)},
                    refit=True, scoring='roc_auc',
                    n_jobs=-1, verbose=0,
                    pre_dispatch="n_jobs/8",
                    cv=3)

gsa.fit(X_train, y_train, sample_weight=weights_train)

print(gsa.best_params_)

# In[ ]:

# In fact, splitting in train/test with a Splitter for another set of datas is useless but it is kept for simplicity (and coherence with Splitter.py). To optimize the speed of the code, it normally has to be changed. However, it works. 

with np.load(exp_dir_2+'traintest.npz') as data:
    X_train_2 = data['X_train'][:nevents_train, :ncomp]
    X_test_2 = data['X_test'][:nevents_test, :ncomp]
    y_train_2 = data['y_train'][:nevents_train]
    y_test_2 = data['y_test'][:nevents_test]
    weights_train_2 = data['weights_train'][:nevents_train]
    weights_test_2 = data['weights_test'][:nevents_test]
    evid_train_2 = data['evid_train'][:nevents_train]
    evid_test_2 = data['evid_test'][:nevents_test]
    
les_scores_train = gsa.best_estimator_.predict_proba(X_train_2)[:,1]

print(len(les_scores_train))

les_scores_test = gsa.best_estimator_.predict_proba(X_test_2)[:,1]

print(len(les_scores_test))

les_scores=np.concatenate((les_scores_train,les_scores_test),axis=None)

print(len(les_scores))

evid=np.concatenate((evid_train_2,evid_test_2),axis=None)

datas = {'EventID': evid, 'ScoresOPU': les_scores}

# In[ ]:


scores_avec_eventid=pd.DataFrame(datas)
scores_avec_eventid.to_csv('/data/atlas/struillou/StageM1VictorStruillou/ScoresOPU_ttbar/ScoresOPU.csv',index=False)

# In[ ]:

#accuracy_train=gsa.score(X_train,y_train, sample_weight=weights_train)
#accuracy_test=gsa.score(X_test,y_test, sample_weight=weights_test)

#print('accuracy train:', accuracy_train)
#print('accuracy test:', accuracy_test)


# In[ ]:


y_pred = gsa.best_estimator_.predict_proba(X_test)[:,1]
y_pred = y_pred.ravel()
y_pred_train = gsa.best_estimator_.predict_proba(X_train)[:,1].ravel()
auc_test = roc_auc_score(y_true=y_test, y_score=y_pred, sample_weight=weights_test)
print("auc test:",auc_test)
print ("auc train:",roc_auc_score(y_true=y_train, y_score=y_pred_train, sample_weight=weights_train))


# In[ ]:


from math import sqrt
from math import log
def amsasimov(s,b): # asimov significance arXiv:1007.1727 eq. 97
        if b<=0 or s<=0:
            return 0
        try:
            return sqrt(2*((s+b)*log(1+float(s)/b)-s))
        except ValueError:
            print(1+float(s)/b)
            print (2*((s+b)*log(1+float(s)/b)-s))
        #return s/sqrt(s+b)


# In[ ]:


def compare_train_test(y_pred_train, y_train, y_pred, y_test, high_low=(0,1), bins=30, xlabel="", ylabel="Arbitrary units", title="", weights_train=np.array([]), weights_test=np.array([])):
    if weights_train.size != 0:
        weights_train_signal = weights_train[y_train == 1]
        weights_train_background = weights_train[y_train == 0]
    else:
        weights_train_signal = None
        weights_train_background = None
    plt.hist(y_pred_train[y_train == 1],
                 color='r', alpha=0.5, range=high_low, bins=bins,
                 histtype='stepfilled', density=True,
                 label='S (train)', weights=weights_train_signal) # alpha is transparancy
    plt.hist(y_pred_train[y_train == 0],
                 color='b', alpha=0.5, range=high_low, bins=bins,
                 histtype='stepfilled', density=True,
                 label='B (train)', weights=weights_train_background)

    if weights_test.size != 0:
        weights_test_signal = weights_test[y_test == 1]
        weights_test_background = weights_test[y_test == 0]
    else:
        weights_test_signal = None
        weights_test_background = None
    hist, bins = np.histogram(y_pred[y_test == 1],
                                  bins=bins, range=high_low, density=True, weights=weights_test_signal)
    scale = len(y_pred[y_test == 1]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    #width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(y_pred[y_test == 0],
                                  bins=bins, range=high_low, density=True, weights=weights_test_background)
    scale = len(y_pred[y_test == 0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    #width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')


# In[ ]:


plt.close()
compare_train_test(y_pred_train, y_train, y_pred, y_test, xlabel="Score", title="Signaux=W",weights_train=weights_train, weights_test=weights_test)
plt.savefig('Fonction_de_répartition_Modèle_Ensemble_OPU_SGD_W_20000_evts_test_1.pdf')
plt.show()


# In[ ]:

plt.close()
lw = 2
fpr,tpr,_ = roc_curve(y_true=y_test, y_score=y_pred,sample_weight=weights_test)
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='W (AUC  = {})'.format(np.round(auc_test,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Curve_Modèle_Ensemble_OPU_SGD_W_20000_evts_test_1.pdf')
plt.show()
