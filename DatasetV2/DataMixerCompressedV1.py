# Description: see DataMixed

def melanger2(chemin,liste_chemins,taille=1000,nb_max_events=-1):#chemin(str) is the path to the directory where the file with the mixed datas
    #will be and liste_chemins=[qcd,ttbar,W] are the paths where datas are (sorted by reaction type).
    #taille sets the number of events by created file. 
    #nb_max_events sets the total number of events in the created files.
    
    hLF=[]
    labels=[]
    particles=[]
    
    liste_fichiers_h5=[os.listdir(liste_chemins[0]),os.listdir(liste_chemins[1]),os.listdir(liste_chemins[2])]
    liste_nb = [len(liste_fichiers_h5[0]),len(liste_fichiers_h5[1]),len(liste_fichiers_h5[2])]
    liste_counters=[0,0,0]
    liste_au_debut=[True,True,True]
    
    iterateur_decoupage=0
    iterateur_nom=0
    iterateur_events=0
    
    liste_minicounters=[0,0,0]
    liste_typefichier=[0,1,2]
    liste_labels=[[1,0,0],[0,1,0],[0,0,1]]
    
    while liste_counters[0] <= liste_nb[0] or liste_counters[1] <= liste_nb[1] or liste_counters[2] <= liste_nb[2]:
        
        if nb_max_events != -1:
            
            if iterateur_events == nb_max_events:
            
                donnees_melangees = h5py.File(chemin+'//'+'mixed_data_'+str(iterateur_nom)+'.h5', 'w') #a h5py file is created
                #with mixed datas
                
                for i in range(len(hLF)):
                    hLF[i].pop(0)
                    for j in range(len(particles[i])):
                        particles[i][j]=list(particles[i][j])
                        particles[i][j].pop(0)
    
                hLFe = donnees_melangees.create_dataset(name='HLF', data=hLF, dtype="f8")
                labelse = donnees_melangees.create_dataset(name='Labels', data=labels, dtype="f8")
                particlese = donnees_melangees.create_dataset(name='Particles', data=particles, dtype="f8")
    
                donnees_melangees.close()
        
                break
        
            iterateur_events = iterateur_events+1
            print(iterateur_events)
        
        if iterateur_decoupage < taille: # able the cutting in many datasets
            
            iterateur_decoupage=iterateur_decoupage+1
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
                    
                    liste_typefichier[choix].close()
                    liste_counters[choix]= liste_counters[choix]+1
                    
                    if liste_counters[choix]< liste_nb[choix]: # ...if there are always files
                    
                        liste_minicounters[choix]=0
                
                        liste_typefichier[choix] = h5py.File(liste_chemins[choix] + "//" + liste_fichiers_h5[choix][liste_counters[choix]], "r")
                
                        hLF.append(list(liste_typefichier[choix]['HLF'][0]))
                        labels.append(liste_labels[choix])
                        particles.append(list(liste_typefichier[choix]['Particles'][0]))
                    
                        liste_minicounters[choix] = liste_minicounters[choix]+1
        
        else:
        
            donnees_melangees = h5py.File(chemin+'//'+'mixed_data_'+str(iterateur_nom)+'.h5', 'w') #a h5py file is created
            #with mixed datas
            
            for i in range(len(hLF)):
                hLF[i].pop(0)
                for j in range(len(particles[i])):
                    particles[i][j]=list(particles[i][j])
                    particles[i][j].pop(0)
    
            hLFe = donnees_melangees.create_dataset(name='HLF', data=hLF, dtype="f8")
            labelse = donnees_melangees.create_dataset(name='Labels', data=labels, dtype="f8")
            particlese = donnees_melangees.create_dataset(name='Particles', data=particles, dtype="f8")
    
            donnees_melangees.close()
        
            hLF=[]
            labels=[]
            particles=[]
        
            iterateur_decoupage=0
            iterateur_nom=iterateur_nom+1
