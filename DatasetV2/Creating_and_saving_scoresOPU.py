#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, SGDClassifier
from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


# Type considered, should be ttbar or W
process_type = 'ttbar'
# Repository containing the outputs of Splitter.py
exp_dir = 'C:/Users/vicru/Desktop/StageRousseauM1/PythonML/Megatestnewbase/SortieSplitter/'
# Number of events and of random projections to be used for the ROC curves. They must be smaller or equal to the number
# of events and random features produced by opu_output_analysis.
nevents = 100000
ncomp = 100000
taille_test=0.2


# In[ ]:


nevents_train=int((1-taille_test)*nevents)
nevents_test=int(taille_test*nevents)


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


clh = SGDClassifier(max_iter=100, loss="log")
gsa = GridSearchCV(clh, 
                   param_grid={"alpha": np.logspace(-3, 2, num=20)}, 
                   n_jobs=-1, verbose=2, cv=3)
gsa.fit(X_train, y_train)


# In[ ]:


les_scores_train = gsa.predict_proba(X_train)[:,1]
les_scores_test = gsa.predict_proba(X_test)[:,1]
les_scores=np.concatenate((les_scores_train,les_scores_test),axis=None)
evid=np.concatenate((evid_train,evid_test),axis=None)

datas = {'EventID': evid, 'ScoresOPU': les_scores}


# In[ ]:


scores_avec_eventid=pd.DataFrame(datas)
scores_avec_eventid.to_csv('C:/Users/vicru/Desktop/StageRousseauM1/PythonML/Megatestnewbase/ScoresAvecEvtID/ScoresOPU.csv',index=False)


# In[ ]:


#Ici on suppose qu'on discrimine ttbar


# In[ ]:


accuracy_train=gsa.score(X_train,y_train, sample_weight=weights_train)
accuracy_test=gsa.score(X_test,y_test, sample_weight=weights_test)

print(accuracy_train)
print(accuracy_test)


# In[ ]:


y_pred = gsa.predict_proba(X_test)[:,1]
y_pred = y_pred.ravel()
y_pred_train = gsa.predict_proba(X_train)[:,1].ravel()
auc_test = roc_auc_score(y_true=y_test, y_score=y_pred, sample_weight=weights_test.values)
print("auc test:",auc_test)
print ("auc train:",roc_auc_score(y_true=y_train, y_score=y_pred_train, sample_weight=weights_train.values))


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
compare_train_test(y_pred_train, y_train, y_pred, y_test, xlabel="Score", title="Signaux=ttbar",weights_train=weights_train.values, weights_test=weights_test.values)
plt.show()


# In[ ]:


lw = 2
fpr,tpr,_ = roc_curve(y_true=y_test, y_score=y_pred,sample_weight=weights_test.values)
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ttbar (AUC  = {})'.format(np.round(auc_test,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#Ici on suppose qu'on discrimine W


# In[ ]:


#idem que ci-dessus mais c'est avec W

