#This code takes the outputs of DatamixerWithEvtID.py and those of Creating_and_saving_scores.py and generate a BDT with them.

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

def creation_megadatasets(chemin):
    
    features = [ 'EventID'
        ,'HT', 'MET', 'PhiMET', 'MT', 'nJets', 'bJets','LepPt', 'LepEta', 'LepPhi', 'LepIsoCh', 'LepIsoGamma','LepIsoNeu', 'LepCharge', 'LepIsEle']

    fichiers_mixes=os.listdir(chemin)
    nb_fichiers=len(fichiers_mixes)
    
    hLF=[]
    labels=[]
    weights=[]
    
    for i in range(nb_fichiers):
        
        un_fichier = h5py.File(chemin + "//" + fichiers_mixes[i], "r")
        
        datasetHLF=un_fichier["HLF"]
        datasetLabels=un_fichier["Labels"]
        
        datasetHLF=datasetHLF[:]
        datasetLabels=datasetLabels[:]
        
        for j in range(len(datasetHLF)):
        
            hLF.append(list(datasetHLF)[j])
            
            if list(datasetLabels[j])==[0.,1.,0.]:
                labels.append(1)
            else:
                labels.append(0)
            
            if list(datasetLabels[j])==[0.,1.,0.]:
                weights.append(0.003)
                
            elif list(datasetLabels[j])==[1.,0.,0.]:
                weights.append(0.362)
                
            else:
                weights.append(0.635)
        
        un_fichier.close()
        
    hLF=pd.DataFrame(hLF,columns=features)
    labels=pd.DataFrame(labels,columns=['label'])['label']
    weights=pd.DataFrame(weights,columns=['weights'])
    
    scoresOPU=pd.read_csv('C:/Users/vicru/Desktop/StageRousseauM1/PythonML/Megatestnewbase/ScoresAvecEvtID/ScoresOPU.csv') #ScoresOPU
    hLF=hLF.merge(scoresOPU,on='EventID') #Merging on EventID criteria
    
    hlf.drop(column = 'EventID')
    
    return hLF,labels,weights
    
##
    
datasetHLF,target,weights = creation_megadatasets('C:/Users/vicru/Desktop/StageRousseauM1/PythonML/Megatestnewbase/Melange')

x_train,x_test,y_train,y_test,weights_train,weights_test = train_test_split(datasetHLF,target,weights,test_size=0.3)
modelTree=lgb.LGBMClassifier()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

modelTree.fit(x_train,y_train, #sample_weights=weights_train.values[:,0]
             )

accuracytest = modelTree.score(x_test,y_test)
accuracytrain = modelTree.score(x_train,y_train)

print(accuracytest)
print(accuracytrain)

##

y_pred_gbm = modelTree.predict_proba(x_test)[:,1]
y_pred_gbm = y_pred_gbm.ravel()
y_pred_train_gbm = modelTree.predict_proba(x_train)[:,1].ravel()
auc_test_gbm = roc_auc_score(y_true=y_test, y_score=y_pred_gbm)
print("auc test:",auc_test_gbm)
print ("auc train:",roc_auc_score(y_true=y_train, y_score=y_pred_train_gbm,))

##

def amsasimov(s,b): # asimov significance arXiv:1007.1727 eq. 97
        if b<=0 or s<=0:
            return 0
        try:
            return sqrt(2*((s+b)*log(1+float(s)/b)-s))
        except ValueError:
            print(1+float(s)/b)
            print (2*((s+b)*log(1+float(s)/b)-s))
        #return s/sqrt(s+b)
        
##
        
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
    
##
    
plt.close()
compare_train_test(y_pred_train_gbm, y_train, y_pred_gbm, y_test, xlabel="LightGBM score", title="LightGBM", #weights_train=weights_train.values[:,0]
                   ,weights_test=weights_test.values[:,0])
plt.savefig("Score_BDT_LightGBM.pdf")
plt.show()

lw = 2
fpr_gbm,tpr_gbm,_ = roc_curve(y_true=y_test, y_score=y_pred_gbm,sample_weight=weights_test.values[:,0])
plt.plot(fpr_gbm, tpr_gbm, color='darkorange',lw=lw, label='LightGBM (AUC  = {})'.format(np.round(auc_test_gbm,decimals=2)))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
#import os
#new_dir = "Plots/Comparing" 
#if not os.path.isdir(new_dir):
#    os.mkdir(new_dir)
plt.savefig("ROC_comparing.pdf")
plt.show()

##

plt.figure()

ax=datasetHLF[target==0].hist(weights=weights[target==0],figsize=(15,12),color='b',alpha=0.5,density=True,label="B")
ax=ax.flatten()[:datasetHLF.shape[1]] # to avoid error if holes in the grid of plots (like if 7 or 8 features)
datasetHLF[target==1].hist(weights=weights[target==1],figsize=(15,12),color='r',alpha=0.5,density=True,ax=ax,label="S")

plt.legend(loc="best")
plt.show()

##

plt.close()

import seaborn as sn # seaborn for nice plot quicker
print ("Background feature correlation matrix")
corrMatrix = datasetHLF[target==0].corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

##

print ("Signal feature correlation matrix")
corrMatrix = datasetHLF[target==1].corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

##

plt.bar(datasetHLF.columns.values, modelTree.feature_importances_)
plt.xticks(rotation=90)
plt.title("Feature importances LightGBM")
#plt.savefig(new_dir + "/VarImp_BDT_LightGBM.pdf",bbox_inches='tight')
plt.show()

##

PI_gbm = PermulationImportance(model=modelTree, X=x_test,y=y_test,weights=weights_test.values[:,0],n_iterations=1,usePredict_poba=True, scoreFunction="amsasimov", colNames=list(features.columns.values))
#PI_gbm.dislayResults()
plott_gbm = PI_gbm.plotBars()
plt.xticks(rotation=90)
plott_gbm.show()
