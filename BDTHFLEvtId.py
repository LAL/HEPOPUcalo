#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.metrics import roc_curve
from permutationimportancephysics.PermutationImportance import PermulationImportance


# In[2]:


def creation_megadatasets(chemin,limit):
    
    features = ['EventID','HT', 'MET', 'PhiMET', 'MT', 'nJets', 'bJets','LepPt', 'LepEta', 'LepPhi', 'LepIsoCh', 'LepIsoGamma','LepIsoNeu', 'LepCharge', 'LepIsEle']

    fichiers_mixes=os.listdir(chemin)
    nb_fichiers=len(fichiers_mixes)
    
    hLF=[]
    labels_signaux_ttbar=[]
    labels_signaux_W=[]
    weights=[]
    
    compteur=0
    stop=False
    
    for i in range(nb_fichiers):
        
        if stop:
            break
        
        un_fichier = h5py.File(chemin + "//" + fichiers_mixes[i], "r")
        
        datasetHLF=un_fichier["HLF"]
        datasetLabels=un_fichier["Labels"]
        
        datasetHLF=datasetHLF[:]
        datasetLabels=datasetLabels[:]
        
        for j in range(len(datasetHLF)):
            
            if limit==compteur:
                stop=True
                break
            
            compteur=compteur+1
        
            hLF.append(list(datasetHLF)[j])
            
            if list(datasetLabels[j])==[0.,1.,0.]:
                labels_signaux_ttbar.append(1)
            else:
                labels_signaux_ttbar.append(0)
                
            if list(datasetLabels[j])==[0.,0.,1.]:
                labels_signaux_W.append(1)
            else:
                labels_signaux_W.append(0)
                
            if list(datasetLabels[j])==[0.,1.,0.]:
                weights.append(0.003)
                
            elif list(datasetLabels[j])==[1.,0.,0.]:
                weights.append(0.362)
                
            else:
                weights.append(0.635)
        
        un_fichier.close()
        
    hLF=pd.DataFrame(hLF,columns=features)
    labels_signaux_ttbar=pd.DataFrame(labels_signaux_ttbar,columns=['label'])['label']
    labels_signaux_W=pd.DataFrame(labels_signaux_W,columns=['label'])['label']
    weights=pd.DataFrame(weights,columns=['weights'])
    
    scoresOPU=pd.read_csv('/data/atlas/struillou/StageM1VictorStruillou/ScoresOPU_W/ScoresOPU.csv')
    
    print(scoresOPU)
    
    hLF_new=hLF.copy(deep=True)
    
    hLF_new=hLF_new.merge(scoresOPU,on='EventID')
    
    print(hLF)
    
    print(hLF_new)
    
    return hLF,hLF_new,labels_signaux_ttbar,labels_signaux_W,weights
        


# In[3]:


datasetHLF,datasetHLF_new,labels_signaux_ttbar,labels_signaux_W,weights = creation_megadatasets('/data/atlas/struillou/StageM1VictorStruillou/SortieDatamixer_2',100000)


# In[4]:

train_size=0.8

hLF = datasetHLF.drop(['EventID'],axis='columns')
hLF_new = datasetHLF_new.drop(['EventID'],axis='columns')


# In[5]:


hLF


# In[ ]:


hLF_new


# In[6]:


labels_signaux_ttbar


# In[7]:


labels_signaux_W


# In[8]:


weights


# In[30]:


x_train_signaux_ttbar,x_test_signaux_ttbar,y_train_signaux_ttbar,y_test_signaux_ttbar,weights_train_signaux_ttbar,weights_test_signaux_ttbar = train_test_split(hLF,labels_signaux_ttbar,weights,test_size=0.2,stratify=labels_signaux_ttbar)
x_train_signaux_W,x_test_signaux_W,y_train_signaux_W,y_test_signaux_W,weights_train_signaux_W,weights_test_signaux_W = train_test_split(hLF,labels_signaux_W,weights,test_size=0.2,stratify=labels_signaux_W)

class_weights_train_ttbar = (weights_train_signaux_ttbar[y_train_signaux_ttbar == 0].values[:,0].sum(), weights_train_signaux_ttbar[y_train_signaux_ttbar == 1].values[:,0].sum())

for i in range(len(class_weights_train_ttbar)):
    #training dataset: equalize number of background and signal
    print(class_weights_train_ttbar)
    weights_train_signaux_ttbar[y_train_signaux_ttbar == i] *= max(class_weights_train_ttbar)/ class_weights_train_ttbar[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test_signaux_ttbar[y_test_signaux_ttbar == i] *= 1/(1-train_size)
    
class_weights_train_W = (weights_train_signaux_W[y_train_signaux_W == 0].values[:,0].sum(), weights_train_signaux_W[y_train_signaux_W == 1].values[:,0].sum())

for i in range(len(class_weights_train_W)):
    #training dataset: equalize number of background and signal
    weights_train_signaux_W[y_train_signaux_W == i] *= max(class_weights_train_W)/ class_weights_train_W[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test_signaux_W[y_test_signaux_W == i] *= 1/(1-train_size) 

modelTree_ttbar=lgb.LGBMClassifier()
modelTree_W=lgb.LGBMClassifier()

scaler_ttbar = StandardScaler()
scaler_W = StandardScaler()

x_train_signaux_ttbar = scaler_ttbar.fit_transform(x_train_signaux_ttbar)
x_test_signaux_ttbar = scaler_ttbar.transform(x_test_signaux_ttbar)

x_train_signaux_W = scaler_W.fit_transform(x_train_signaux_W)
x_test_signaux_W = scaler_W.transform(x_test_signaux_W)

modelTree_ttbar.fit(x_train_signaux_ttbar,y_train_signaux_ttbar)
modelTree_W.fit(x_train_signaux_W,y_train_signaux_W)

accuracytest_ttbar = modelTree_ttbar.score(x_test_signaux_ttbar,y_test_signaux_ttbar,sample_weight=weights_test_signaux_ttbar.values[:,0])
accuracytrain_ttbar = modelTree_ttbar.score(x_train_signaux_ttbar,y_train_signaux_ttbar,sample_weight=weights_train_signaux_ttbar.values[:,0])

accuracytest_W = modelTree_W.score(x_test_signaux_W,y_test_signaux_W,sample_weight=weights_test_signaux_W.values[:,0])
accuracytrain_W = modelTree_W.score(x_train_signaux_W,y_train_signaux_W,sample_weight=weights_train_signaux_W.values[:,0])

print("accuracytest ttbar:", accuracytest_ttbar)
print("accuracytrain ttbar:", accuracytrain_ttbar)

print("accuracytest W:", accuracytest_W)
print("accuracytrain W:", accuracytrain_W)


# In[35]:


y_pred_ttbar = modelTree_ttbar.predict_proba(x_test_signaux_ttbar)[:,1]
y_pred_ttbar = y_pred_ttbar.ravel()
y_pred_train_ttbar = modelTree_ttbar.predict_proba(x_train_signaux_ttbar)[:,1].ravel()
auc_test_ttbar = roc_auc_score(y_true=y_test_signaux_ttbar, y_score=y_pred_ttbar, sample_weight=weights_test_signaux_ttbar.values[:,0])
print("auc test ttbar:",auc_test_ttbar)
print ("auc train ttbar:",roc_auc_score(y_true=y_train_signaux_ttbar, y_score=y_pred_train_ttbar, sample_weight=weights_train_signaux_ttbar.values[:,0]))


# In[37]:


y_pred_W = modelTree_W.predict_proba(x_test_signaux_W)[:,1]
y_pred_W = y_pred_W.ravel()
y_pred_train_W = modelTree_W.predict_proba(x_train_signaux_W)[:,1].ravel()
auc_test_W = roc_auc_score(y_true=y_test_signaux_W, y_score=y_pred_W,sample_weight=weights_test_signaux_W.values[:,0])
print("auc test W:",auc_test_W)
print ("auc train W:",roc_auc_score(y_true=y_train_signaux_W, y_score=y_pred_train_W,sample_weight=weights_train_signaux_W.values[:,0]))


# In[38]:


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


# In[39]:


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


# In[40]:


plt.close()
compare_train_test(y_pred_train_ttbar, y_train_signaux_ttbar, y_pred_ttbar, y_test_signaux_ttbar, xlabel="Score", title="Signaux=ttbar",weights_train=weights_train_signaux_ttbar.values[:,0], weights_test=weights_test_signaux_ttbar.values[:,0])
plt.savefig('Fonction_de_répartition_Modèle_BDTHLF_ttbar.pdf')
plt.show()


# In[41]:


plt.close()
compare_train_test(y_pred_train_W, y_train_signaux_W, y_pred_W, y_test_signaux_W, xlabel="Score", title="Signaux=ttbar",weights_train=weights_train_signaux_W.values[:,0], weights_test=weights_test_signaux_W.values[:,0])
plt.savefig('Fonction_de_répartition_Modèle_BDTHLF_W.pdf')
plt.show()


# In[42]:

plt.close()
lw = 2
fpr_ttbar,tpr_ttbar,_ = roc_curve(y_true=y_test_signaux_ttbar, y_score=y_pred_ttbar,sample_weight=weights_test_signaux_ttbar.values[:,0])
plt.plot(fpr_ttbar, tpr_ttbar, color='darkorange',lw=lw, label='ttbar (AUC  = {})'.format(np.round(auc_test_ttbar,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèle_BDTHLF_ttbar.pdf')
plt.show()


# In[43]:

plt.close()
lw = 2
fpr_W,tpr_W,_ = roc_curve(y_true=y_test_signaux_W, y_score=y_pred_W,sample_weight=weights_test_signaux_W.values[:,0])
plt.plot(fpr_W, tpr_W, color='darkorange',lw=lw, label='W (AUC  = {})'.format(np.round(auc_test_W,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèle_BDTHLF_W.pdf')
plt.show()


# In[90]:

plt.close()
lw = 2
fpr_ttbar,tpr_ttbar,_ = roc_curve(y_true=y_test_signaux_ttbar, y_score=y_pred_ttbar,sample_weight=weights_test_signaux_ttbar.values[:,0])
plt.plot(fpr_ttbar, tpr_ttbar, color='darkorange',lw=lw, label='ttbar (AUC  = {})'.format(np.round(auc_test_ttbar,decimals=2)))
fpr_W,tpr_W,_ = roc_curve(y_true=y_test_signaux_W, y_score=y_pred_W,sample_weight=weights_test_signaux_W.values[:,0])
plt.plot(fpr_W, tpr_W, color='red',lw=lw, label='W (AUC  = {})'.format(np.round(auc_test_W,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.2])
plt.ylim([0.5, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_ComparaisonWttbar_Modèle_BDTHLF.pdf')
plt.show()

# In[53]:


plt.close()

import seaborn as sn # seaborn for nice plot quicker
print ("Background feature correlation matrix ttbar")
corrMatrix = hLF[labels_signaux_ttbar==0].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Bruit_Modèle_BDTHLF_ttbar.pdf')
plt.show()

plt.close()

print ("Signal feature correlation matrix ttbar")
corrMatrix = hLF[labels_signaux_ttbar==1].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Signal_Modèle_BDTHLF_ttbar.pdf')
plt.show()


# In[52]:


plt.close()

import seaborn as sn # seaborn for nice plot quicker
print ("Background feature correlation matrix W")
corrMatrix = hLF[labels_signaux_W==0].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Bruit_Modèle_BDTHLF_W.pdf')
plt.show()

plt.close()

print ("Signal feature correlation matrix W")
corrMatrix = hLF[labels_signaux_W==1].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Signal_Modèle_BDTHLF_W.pdf')
plt.show()


# In[54]:

plt.close()


plt.bar(hLF.columns.values, modelTree_ttbar.feature_importances_)
plt.xticks(rotation=90)
plt.title("Features importances ttbar")
plt.savefig('Features_Importances_Modèle_BDTHLF_ttbar.pdf')
plt.show()


# In[80]:

plt.close()

plt.bar(hLF.columns.values, modelTree_W.feature_importances_)
plt.xticks(rotation=90)
plt.title("Features importances W")
plt.savefig('Features_Importances_Modèle_BDTHLF_W.pdf')
plt.show()


# In[56]:

plt.close()

PI_ttbar = PermulationImportance(model=modelTree_ttbar, X=x_test_signaux_ttbar,y=y_test_signaux_ttbar,weights=weights_test_signaux_ttbar.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(hLF.columns.values))
plott_ttbar = PI_ttbar.plotBars()
plt.xticks(rotation=90)
plt.savefig('Permutation_Importance_Modèle_BDTHLF_ttbar.pdf')
plott_ttbar.show()


# In[57]:

plt.close()

PI_W = PermulationImportance(model=modelTree_W, X=x_test_signaux_W,y=y_test_signaux_W,weights=weights_test_signaux_W.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(hLF.columns.values))
plott_W = PI_W.plotBars()
plt.xticks(rotation=90)
plt.savefig('Permutation_Importance_Modèle_BDTHLF_W.pdf')
plott_W.show()

plt.close()


# In[ ]:


#deltaphi


# In[58]:


DeltaPhi=(hLF['PhiMET']-hLF['LepPhi'])%(2*np.pi)
DeltaPhi=abs(DeltaPhi-np.pi)

hLF2 = hLF.copy(deep=True)
hLF2['DeltaPhi']=pd.Series(DeltaPhi)


# In[59]:


hLF2


# In[62]:


x_train_signaux_ttbar_dp,x_test_signaux_ttbar_dp,y_train_signaux_ttbar_dp,y_test_signaux_ttbar_dp,weights_train_signaux_ttbar_dp,weights_test_signaux_ttbar_dp = train_test_split(hLF2,labels_signaux_ttbar,weights,test_size=0.2,stratify=labels_signaux_ttbar)
x_train_signaux_W_dp,x_test_signaux_W_dp,y_train_signaux_W_dp,y_test_signaux_W_dp,weights_train_signaux_W_dp,weights_test_signaux_W_dp = train_test_split(hLF2,labels_signaux_W,weights,test_size=0.2,stratify=labels_signaux_W)

class_weights_train_ttbar_dp = (weights_train_signaux_ttbar_dp[y_train_signaux_ttbar_dp == 0].values[:,0].sum(), weights_train_signaux_ttbar_dp[y_train_signaux_ttbar_dp == 1].values[:,0].sum())

for i in range(len(class_weights_train_ttbar_dp)):
    #training dataset: equalize number of background and signal
    weights_train_signaux_ttbar_dp[y_train_signaux_ttbar_dp == i] *= max(class_weights_train_ttbar_dp)/ class_weights_train_ttbar_dp[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test_signaux_ttbar_dp[y_test_signaux_ttbar_dp == i] *= 1/(1-train_size)
    
class_weights_train_W_dp = (weights_train_signaux_W_dp[y_train_signaux_W_dp == 0].values[:,0].sum(), weights_train_signaux_W_dp[y_train_signaux_W_dp == 1].values[:,0].sum())

for i in range(len(class_weights_train_W_dp)):
    #training dataset: equalize number of background and signal
    weights_train_signaux_W_dp[y_train_signaux_W_dp == i] *= max(class_weights_train_W_dp)/ class_weights_train_W_dp[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test_signaux_W_dp[y_test_signaux_W_dp == i] *= 1/(1-train_size) 

modelTree_ttbar_dp=lgb.LGBMClassifier()
modelTree_W_dp=lgb.LGBMClassifier()

scaler_ttbar_dp = StandardScaler()
scaler_W_dp = StandardScaler()

x_train_signaux_ttbar_dp = scaler_ttbar_dp.fit_transform(x_train_signaux_ttbar_dp)
x_test_signaux_ttbar_dp = scaler_ttbar_dp.transform(x_test_signaux_ttbar_dp)

x_train_signaux_W_dp = scaler_W_dp.fit_transform(x_train_signaux_W_dp)
x_test_signaux_W_dp = scaler_W_dp.transform(x_test_signaux_W_dp)

modelTree_ttbar_dp.fit(x_train_signaux_ttbar_dp,y_train_signaux_ttbar_dp)
modelTree_W_dp.fit(x_train_signaux_W_dp,y_train_signaux_W_dp)

accuracytest_ttbar_dp = modelTree_ttbar_dp.score(x_test_signaux_ttbar_dp,y_test_signaux_ttbar_dp,sample_weight=weights_test_signaux_ttbar_dp.values[:,0])
accuracytrain_ttbar_dp = modelTree_ttbar_dp.score(x_train_signaux_ttbar_dp,y_train_signaux_ttbar_dp,sample_weight=weights_train_signaux_ttbar_dp.values[:,0])

accuracytest_W_dp = modelTree_W_dp.score(x_test_signaux_W_dp,y_test_signaux_W_dp,sample_weight=weights_test_signaux_W_dp.values[:,0])
accuracytrain_W_dp = modelTree_W_dp.score(x_train_signaux_W_dp,y_train_signaux_W_dp,sample_weight=weights_train_signaux_W_dp.values[:,0])

print("accuracytest ttbar deltaphi",accuracytest_ttbar_dp)
print("accuracytrain ttbar deltaphi",accuracytrain_ttbar_dp)

print("accuracytest W deltaphi",accuracytest_W_dp)
print("accuracytrain W deltaphi",accuracytrain_W_dp)


# In[63]:


y_pred_ttbar_dp = modelTree_ttbar_dp.predict_proba(x_test_signaux_ttbar_dp)[:,1]
y_pred_ttbar_dp = y_pred_ttbar_dp.ravel()
y_pred_train_ttbar_dp = modelTree_ttbar_dp.predict_proba(x_train_signaux_ttbar_dp)[:,1].ravel()
auc_test_ttbar_dp = roc_auc_score(y_true=y_test_signaux_ttbar_dp, y_score=y_pred_ttbar_dp, sample_weight=weights_test_signaux_ttbar_dp.values[:,0])
print("auc test ttbar avec deltaphi:",auc_test_ttbar_dp)
print ("auc train ttbar avec deltaphi:",roc_auc_score(y_true=y_train_signaux_ttbar_dp, y_score=y_pred_train_ttbar_dp, sample_weight=weights_train_signaux_ttbar_dp.values[:,0]))


# In[64]:


y_pred_W_dp = modelTree_W_dp.predict_proba(x_test_signaux_W_dp)[:,1]
y_pred_W_dp = y_pred_W_dp.ravel()
y_pred_train_W_dp = modelTree_W_dp.predict_proba(x_train_signaux_W_dp)[:,1].ravel()
auc_test_W_dp = roc_auc_score(y_true=y_test_signaux_W_dp, y_score=y_pred_W_dp,sample_weight=weights_test_signaux_W_dp.values[:,0])
print("auc test W avec deltaphi:",auc_test_W_dp)
print ("auc train W avec deltaphi:",roc_auc_score(y_true=y_train_signaux_W_dp, y_score=y_pred_train_W_dp,sample_weight=weights_train_signaux_W_dp.values[:,0]))


# In[65]:


plt.close()
compare_train_test(y_pred_train_ttbar_dp, y_train_signaux_ttbar_dp, y_pred_ttbar_dp, y_test_signaux_ttbar_dp, xlabel="Score", title="Signaux=ttbar (avec deltaphi)",weights_train=weights_train_signaux_ttbar_dp.values[:,0], weights_test=weights_test_signaux_ttbar_dp.values[:,0])
plt.savefig('Fonction_de_répartition_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plt.show()


# In[66]:


plt.close()
compare_train_test(y_pred_train_W_dp, y_train_signaux_W_dp, y_pred_W_dp, y_test_signaux_W_dp, xlabel="Score", title="Signaux=ttbar (avec deltaphi)",weights_train=weights_train_signaux_W_dp.values[:,0], weights_test=weights_test_signaux_W_dp.values[:,0])
plt.savefig('Fonction_de_répartition_Modèle_BDTHLF_W_deltaphi.pdf')
plt.show()


# In[67]:

plt.close()

lw = 2
fpr_ttbar_dp,tpr_ttbar_dp,_ = roc_curve(y_true=y_test_signaux_ttbar_dp, y_score=y_pred_ttbar_dp,sample_weight=weights_test_signaux_ttbar_dp.values[:,0])
plt.plot(fpr_ttbar_dp, tpr_ttbar_dp, color='darkorange',lw=lw, label='ttbar avec deltaphi (AUC  = {})'.format(np.round(auc_test_ttbar_dp,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plt.show()


# In[68]:

plt.close()

lw = 2
fpr_W_dp,tpr_W_dp,_ = roc_curve(y_true=y_test_signaux_W_dp, y_score=y_pred_W_dp,sample_weight=weights_test_signaux_W_dp.values[:,0])
plt.plot(fpr_W_dp, tpr_W_dp, color='darkorange',lw=lw, label='W avec deltaphi (AUC  = {})'.format(np.round(auc_test_W_dp,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèle_BDTHLF_W_deltaphi.pdf')
plt.show()


# In[73]:


plt.close()

lw = 2
fpr_ttbar_dp,tpr_ttbar_dp,_ = roc_curve(y_true=y_test_signaux_ttbar_dp, y_score=y_pred_ttbar_dp,sample_weight=weights_test_signaux_ttbar_dp.values[:,0])
plt.plot(fpr_ttbar_dp, tpr_ttbar_dp, color='darkorange',lw=lw, label='ttbar avec deltaphi (AUC  = {})'.format(np.round(auc_test_ttbar_dp,decimals=2)))

fpr_ttbar,tpr_ttbar,_ = roc_curve(y_true=y_test_signaux_ttbar, y_score=y_pred_ttbar,sample_weight=weights_test_signaux_ttbar.values[:,0])
plt.plot(fpr_ttbar, tpr_ttbar, color='red',lw=lw, label='ttbar (AUC  = {})'.format(np.round(auc_test_ttbar,decimals=2)))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.15])
plt.ylim([0.8, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèles_BDTHLF_ttbar_vs_ttbaravecdeltaphi.pdf')
plt.show()


# In[74]:


plt.close()

lw = 2
fpr_W_dp,tpr_W_dp,_ = roc_curve(y_true=y_test_signaux_W_dp, y_score=y_pred_W_dp,sample_weight=weights_test_signaux_W_dp.values[:,0])
plt.plot(fpr_W_dp, tpr_W_dp, color='darkorange',lw=lw, label='W avec deltaphi (AUC  = {})'.format(np.round(auc_test_W_dp,decimals=2)))

fpr_W,tpr_W,_ = roc_curve(y_true=y_test_signaux_W, y_score=y_pred_W,sample_weight=weights_test_signaux_W.values[:,0])
plt.plot(fpr_W, tpr_W, color='red',lw=lw, label='W (AUC  = {})'.format(np.round(auc_test_W,decimals=2)))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.2])
plt.ylim([0.6, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèles_BDTHLF_W_vs_Wavecdeltaphi.pdf')
plt.show()

# In[77]:


plt.close()

import seaborn as sn # seaborn for nice plot quicker
print ("Background feature correlation matrix ttbar avec deltaphi")
corrMatrix = hLF2[labels_signaux_ttbar==0].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Signal_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plt.show()

plt.close()

print ("Signal feature correlation matrix ttbar avec deltaphi")
corrMatrix = hLF2[labels_signaux_ttbar==1].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Bruit_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plt.show()


# In[78]:


plt.close()
import seaborn as sn # seaborn for nice plot quicker
print ("Background feature correlation matrix W avec deltaphi")
corrMatrix = hLF2[labels_signaux_W==0].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Signal_Modèle_BDTHLF_W_deltaphi.pdf')
plt.show()

plt.close()
print ("Signal feature correlation matrix W avec deltaphi")
corrMatrix = hLF2[labels_signaux_W==1].corr()
sn.heatmap(corrMatrix, annot=False)
plt.savefig('Correlation_Matrix_Bruit_Modèle_BDTHLF_W_deltaphi.pdf')
plt.show()


# In[79]:

plt.close()

plt.bar(hLF2.columns.values, modelTree_ttbar_dp.feature_importances_)
plt.xticks(rotation=90)
plt.title("Features importances ttbar avec deltaphi")
plt.savefig('Features_Importances_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plt.show()


# In[81]:

plt.close()

plt.bar(hLF2.columns.values, modelTree_W_dp.feature_importances_)
plt.xticks(rotation=90)
plt.title("Features importances W avec deltaphi")
plt.savefig('Features_Importances_Modèle_BDTHLF_W_deltaphi.pdf')
plt.show()


# In[82]:

plt.close()

PI_ttbar_dp = PermulationImportance(model=modelTree_ttbar_dp, X=x_test_signaux_ttbar_dp,y=y_test_signaux_ttbar_dp,weights=weights_test_signaux_ttbar_dp.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(hLF2.columns.values))
plott_ttbar_dp = PI_ttbar_dp.plotBars()
plt.xticks(rotation=90)
plt.savefig('Permutation_Importance_Modèle_BDTHLF_ttbar_deltaphi.pdf')
plott_ttbar_dp.show()


# In[83]:

plt.close()

PI_W_dp = PermulationImportance(model=modelTree_W_dp, X=x_test_signaux_W_dp,y=y_test_signaux_W_dp,weights=weights_test_signaux_W_dp.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(hLF2.columns.values))
plott_W_dp = PI_W_dp.plotBars()
plt.xticks(rotation=90)
plt.savefig('Permutation_Importance_Modèle_BDTHLF_W_deltaphi.pdf')
plott_W_dp.show()


# In[1]:


#Travail sur hLF_new
#Dans cette partie, on suppose que les ScoresOPU correspondent à une discrimination:
#signal= ttbar et bruit = W + QCD
#On travaille également avec deltaphi


# In[ ]:

print(hLF_new)

DeltaPhi=(hLF_new['PhiMET']-hLF_new['LepPhi'])%(2*np.pi)
DeltaPhi=abs(DeltaPhi-np.pi)

hLF_new_deltaphi = hLF_new.copy(deep=True)
hLF_new_deltaphi['DeltaPhi']=pd.Series(DeltaPhi)


# In[ ]:


print(hLF_new_deltaphi)


# In[62]:


x_train_signaux_ttbar_dp_new,x_test_signaux_ttbar_dp_new,y_train_signaux_ttbar_dp_new,y_test_signaux_ttbar_dp_new,weights_train_signaux_ttbar_dp_new,weights_test_signaux_ttbar_dp_new = train_test_split(hLF_new_deltaphi,labels_signaux_ttbar,weights,test_size=0.2,stratify=labels_signaux_ttbar)

class_weights_train_ttbar_dp_new = (weights_train_signaux_ttbar_dp_new[y_train_signaux_ttbar_dp_new == 0].values[:,0].sum(), weights_train_signaux_ttbar_dp_new[y_train_signaux_ttbar_dp_new == 1].values[:,0].sum())

for i in range(len(class_weights_train_ttbar_dp_new)):
    #training dataset: equalize number of background and signal
    weights_train_signaux_ttbar_dp_new[y_train_signaux_ttbar_dp_new == i] *= max(class_weights_train_ttbar_dp_new)/ class_weights_train_ttbar_dp_new[i] 
    #test dataset : increase test weight to compensate for sampling
    weights_test_signaux_ttbar_dp_new[y_test_signaux_ttbar_dp_new == i] *= 1/(1-train_size)

modelTree_ttbar_dp_new=lgb.LGBMClassifier()

scaler_ttbar_dp_new = StandardScaler()

x_train_signaux_ttbar_dp_new = scaler_ttbar_dp_new.fit_transform(x_train_signaux_ttbar_dp_new)
x_test_signaux_ttbar_dp_new = scaler_ttbar_dp_new.transform(x_test_signaux_ttbar_dp_new)

modelTree_ttbar_dp_new.fit(x_train_signaux_ttbar_dp_new,y_train_signaux_ttbar_dp_new)

accuracytest_ttbar_dp_new = modelTree_ttbar_dp_new.score(x_test_signaux_ttbar_dp_new,y_test_signaux_ttbar_dp_new,sample_weight=weights_test_signaux_ttbar_dp_new.values[:,0])
accuracytrain_ttbar_dp_new = modelTree_ttbar_dp_new.score(x_train_signaux_ttbar_dp_new,y_train_signaux_ttbar_dp_new,sample_weight=weights_train_signaux_ttbar_dp_new.values[:,0])

print("accuracytest ttbar deltaphi avec scoresOPU:",accuracytest_ttbar_dp_new)
print("accuracytrain ttbar deltaphi avec scoresOPU",accuracytrain_ttbar_dp_new)


# In[63]:


y_pred_ttbar_dp_new = modelTree_ttbar_dp_new.predict_proba(x_test_signaux_ttbar_dp_new)[:,1]
y_pred_ttbar_dp_new = y_pred_ttbar_dp_new.ravel()
y_pred_train_ttbar_dp_new = modelTree_ttbar_dp_new.predict_proba(x_train_signaux_ttbar_dp_new)[:,1].ravel()
auc_test_ttbar_dp_new = roc_auc_score(y_true=y_test_signaux_ttbar_dp_new, y_score=y_pred_ttbar_dp_new, sample_weight=weights_test_signaux_ttbar_dp_new.values[:,0])
print("auc test ttbar avec deltaphi et scores OPU:",auc_test_ttbar_dp_new)
print ("auc train ttbar avec deltaphi et scores OPU:",roc_auc_score(y_true=y_train_signaux_ttbar_dp_new, y_score=y_pred_train_ttbar_dp_new, sample_weight=weights_train_signaux_ttbar_dp_new.values[:,0]))


# In[65]:


plt.close()
compare_train_test(y_pred_train_ttbar_dp_new, y_train_signaux_ttbar_dp_new, y_pred_ttbar_dp_new, y_test_signaux_ttbar_dp_new, xlabel="Score", title="Signaux=ttbar (avec deltaphi et Scores OPU)",weights_train=weights_train_signaux_ttbar_dp_new.values[:,0], weights_test=weights_test_signaux_ttbar_dp_new.values[:,0])
plt.savefig('Fonction_Répartition_Modèle_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plt.show()


# In[67]:

plt.close()
lw = 2
fpr_ttbar_dp_new,tpr_ttbar_dp_new,_ = roc_curve(y_true=y_test_signaux_ttbar_dp_new, y_score=y_pred_ttbar_dp_new,sample_weight=weights_test_signaux_ttbar_dp_new.values[:,0])
plt.plot(fpr_ttbar_dp_new, tpr_ttbar_dp_new, color='darkorange',lw=lw, label='ttbar avec deltaphi et Scores OPU (AUC  = {})'.format(np.round(auc_test_ttbar_dp_new,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Modèle_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plt.show()


# In[73]:


plt.close()

lw = 2

fpr_ttbar_dp_new,tpr_ttbar_dp_new,_ = roc_curve(y_true=y_test_signaux_ttbar_dp_new, y_score=y_pred_ttbar_dp_new,sample_weight=weights_test_signaux_ttbar_dp_new.values[:,0])
plt.plot(fpr_ttbar_dp_new, tpr_ttbar_dp_new, color='blue',lw=lw, label='ttbar avec deltaphi et Scores OPU (AUC  = {})'.format(np.round(auc_test_ttbar_dp_new,decimals=2)))

fpr_ttbar_dp,tpr_ttbar_dp,_ = roc_curve(y_true=y_test_signaux_ttbar_dp, y_score=y_pred_ttbar_dp,sample_weight=weights_test_signaux_ttbar_dp.values[:,0])
plt.plot(fpr_ttbar_dp, tpr_ttbar_dp, color='darkorange',lw=lw, label='ttbar avec deltaphi (AUC  = {})'.format(np.round(auc_test_ttbar_dp,decimals=2)))

fpr_ttbar,tpr_ttbar,_ = roc_curve(y_true=y_test_signaux_ttbar, y_score=y_pred_ttbar,sample_weight=weights_test_signaux_ttbar.values[:,0])
plt.plot(fpr_ttbar, tpr_ttbar, color='red',lw=lw, label='ttbar (AUC  = {})'.format(np.round(auc_test_ttbar,decimals=2)))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.15])
plt.ylim([0.8, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Megacomparaison_Modèles_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plt.show()

# In[75]:

plt.close()
plt.figure()

ax=hLF_new_deltaphi[labels_signaux_ttbar==0].hist(weights=weights[labels_signaux_ttbar==0],figsize=(15,12),color='b',alpha=0.5,density=True,label="B")
ax=ax.flatten()[:hLF_new_deltaphi.shape[1]] # to avoid error if holes in the grid of plots (like if 7 or 8 features)
hLF_new_deltaphi[labels_signaux_ttbar==1].hist(weights=weights[labels_signaux_ttbar==1],figsize=(15,12),color='r',alpha=0.5,density=True,ax=ax,label="S")

plt.legend(loc="best")
plt.savefig('Repartition_Features_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plt.show()

# In[79]:

plt.close()
plt.bar(hLF_new_deltaphi.columns.values, modelTree_ttbar_dp_new.feature_importances_)
plt.xticks(rotation=90)
plt.title("Features importances ttbar avec deltaphi et scores OPU")
plt.savefig('FI_Modèles_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plt.show()
plt.close()

# In[82]:

plt.close()

PI_ttbar_dp_new = PermulationImportance(model=modelTree_ttbar_dp_new, X=x_test_signaux_ttbar_dp_new,y=y_test_signaux_ttbar_dp_new,weights=weights_test_signaux_ttbar_dp_new.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(hLF_new_deltaphi.columns.values))
plott_ttbar_dp_new = PI_ttbar_dp_new.plotBars()
plt.xticks(rotation=90)
plt.savefig('PI_Modèles_BDTHLF_ttbar_deltaphi_scoresOPU.pdf')
plott_ttbar_dp_new.show()
plt.close()


# In[ ]:


#Travail sur hLF_new
#Dans cette partie, on suppose que les ScoresOPU correspondent à une discrimination:
#signal= W et bruit = W + ttbar
#On travaille également avec deltaphi


# In[ ]:


#refaire idem que ci-dessus avec W

# In[76]:

##########################################W

plt.close()

plt.figure()

ax=hLF2[labels_signaux_W==0].hist(weights=weights[labels_signaux_W==0],figsize=(15,12),color='b',alpha=0.5,density=True,label="B")
ax=ax.flatten()[:hLF2.shape[1]] # to avoid error if holes in the grid of plots (like if 7 or 8 features)
hLF2[labels_signaux_W==1].hist(weights=weights[labels_signaux_W==1],figsize=(15,12),color='r',alpha=0.5,density=True,ax=ax,label="S")


plt.legend(loc="best")
plt.savefig('Repartition_Features_BDTHLF_W_deltaphi.pdf')
plt.show()
plt.close()

