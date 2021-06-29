#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import h5py
import pandas as pd
import numpy as np
from scipy import misc
import time
import sys
import matplotlib 
import os
import matplotlib.pyplot as plt
import random
from array import array


# In[2]:


def melanger4(chemin,liste_chemins,taille=1000,nb_max_boucles=-5):#chemin(str) is the path to the directory where the file with the mixed datas
    #will be and liste_chemins=[qcd,ttbar,W] are the paths where datas are (sorted by reaction type).
    #taille sets the number of loops by created file made by the algorithms. 
    #nb_max_boucles sets the total number of loops made by the algorithm.
    
    hLF=[]
    labels=[]
    particles=[]
    
    liste_fichiers_h5=[os.listdir(liste_chemins[0]),os.listdir(liste_chemins[1]),os.listdir(liste_chemins[2])]
    liste_nb = [len(liste_fichiers_h5[0]),len(liste_fichiers_h5[1]),len(liste_fichiers_h5[2])]
    liste_counters=[0,0,0]
    liste_au_debut=[True,True,True]
    liste_iterateurs=[0,0,-1]
    liste_minicounters=[0,0,0]
    liste_typefichier=[0,1,2]
    liste_labels=[[1,0,0],[0,1,0],[0,0,1]]
    
    while liste_counters[0] <= liste_nb[0] or liste_counters[1] <= liste_nb[1] or liste_counters[2] <= liste_nb[2]: # if there always are files...
        
        liste_iterateurs[2] = liste_iterateurs[2]+1
        print(liste_iterateurs[2])
            
        if liste_iterateurs[0]==taille or liste_iterateurs[2]==nb_max_boucles: # the exit files consist in many little files
            
            donnees_melangees = h5py.File(chemin+'//'+'mixed_data_'+str(liste_iterateurs[1])+'.h5', 'w') #a h5py file is created
            #with mixed datas
            
            for i in range(len(hLF)): # modifing the EventID column
                hLF[i][0]=int(i+taille*liste_iterateurs[1])
                for j in range(len(particles[i])):
                    particles[i][j]=list(particles[i][j])
                    particles[i][j][0]=int(i+taille*liste_iterateurs[1])
    
            hLFe = donnees_melangees.create_dataset(name='HLF', data=hLF, dtype="f8")
            labelse = donnees_melangees.create_dataset(name='Labels', data=labels, dtype="f8")
            particlese = donnees_melangees.create_dataset(name='Particles', data=particles, dtype="f8")
    
            donnees_melangees.close()
        
            hLF=[]
            labels=[]
            particles=[]
        
            liste_iterateurs[0]=0
            liste_iterateurs[1]=liste_iterateurs[1]+1
            
            if liste_iterateurs[2]==nb_max_boucles:
                break
        else:
            
            liste_iterateurs[0]=liste_iterateurs[0]+1
            choix =random.randint(0,2) # repertories are choosen randomly 
            
            if liste_au_debut[choix]: # able the initialisation
                
                liste_minicounters[choix]=0
                
                liste_typefichier[choix] = h5py.File(liste_chemins[choix] + "//" + liste_fichiers_h5[choix][liste_counters[choix]], "r")
                
                hLF.append(list(liste_typefichier[choix]['HLF'][0]))
                labels.append(liste_labels[choix])
                particles.append(list(liste_typefichier[choix]['Particles'][0]))
                
                liste_au_debut[choix]=False
                
                liste_minicounters[choix] = liste_minicounters[choix]+1
            
            else: #the cursor is in the middle of the files...
                
                if liste_minicounters[choix] < len(liste_typefichier[choix]['Particles']): #... but not at the end
                
                    hLF.append(list(liste_typefichier[choix]['HLF'][liste_minicounters[choix]]))
                    labels.append(liste_labels[choix])
                    particles.append(list(liste_typefichier[choix]['Particles'][liste_minicounters[choix]]))
                    
                    liste_minicounters[choix] = liste_minicounters[choix]+1
                
                else: # when the cursor is at the end, the current file is closed and the next one is opened...
                   
                    liste_counters[choix]= liste_counters[choix]+1
                    
                    if liste_counters[choix]< liste_nb[choix]: # ...if there are always files
                        
                        liste_typefichier[choix].close()
                        liste_minicounters[choix]=0
                
                        liste_typefichier[choix] = h5py.File(liste_chemins[choix] + "//" + liste_fichiers_h5[choix][liste_counters[choix]], "r")
                
                        hLF.append(list(liste_typefichier[choix]['HLF'][0]))
                        labels.append(liste_labels[choix])
                        particles.append(list(liste_typefichier[choix]['Particles'][0]))
                    
                        liste_minicounters[choix] = liste_minicounters[choix]+1


# In[3]:


melanger4('/data/atlas/struillou/StageM1VictorStruillou/SortieDatamixer_bdt',['/data/atlas/struillou/StageM1VictorStruillou/qcd_bdt','/data/atlas/struillou/StageM1VictorStruillou/ttbar_bdt','/data/atlas/struillou/StageM1VictorStruillou/W_bdt'],1000,110000)


# In[ ]:




