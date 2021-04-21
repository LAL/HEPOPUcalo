# This code takes the datasets qcd, ttbar and W (sorted by type of reaction) provided by the database of the CERN and
# create a bunch of mixed datas that can be used by TransformArrowSparseFloat code.

def melanger(chemin,qcd,ttbar,w,taille=1000,nb_max_events=-1): #chemin(str) is the path to the directory where the file with the mixed datas
    #will be and qcd,ttbar and w are the paths where datas are (sorted by reaction type).
    #taille sets the number of events by created file. 
    #nb_max_events sets the total number of events in the created files. 
    
    hLF=[]
    labels=[]
    particles=[]
    
    fichiers_h5_qcd = os.listdir(qcd)
    fichiers_h5_ttbar = os.listdir(ttbar)
    fichiers_h5_W = os.listdir(w)
    
    nb_qcd= len(fichiers_h5_qcd)
    nb_ttbar= len(fichiers_h5_ttbar)
    nb_W= len(fichiers_h5_W)
    
    counter_qcd=0
    counter_ttbar=0
    counter_W=0
    
    au_debut_qcd = True
    au_debut_ttbar = True
    au_debut_W = True
    
    iterateur_decoupage=0
    iterateur_nom=0
    
    iterateur_events=0
    
    while counter_qcd <= nb_qcd or counter_ttbar <= nb_ttbar or counter_W <= nb_W:
        
        if nb_max_events != -1:
            
            if iterateur_events == nb_max_events:
            
                donnees_melangees = h5py.File(chemin+'//'+'mixed_data_'+str(iterateur_nom)+'.h5', 'w') #a h5py file is created
                #with mixed datas
                
                for i in range(len(hLF)): # removing the EventID feature
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
        
        if iterateur_decoupage <= taille: # able the cutting in many datasets
            
            iterateur_decoupage=iterateur_decoupage+1
            choix =random.randint(1,3) # repertories are choosen randomly
        
            if choix == 1 : #qcd was choosen
            
                if au_debut_qcd: # able the initialisation
                
                    minicounter_qcd=0
                
                    un_fichier_qcd = h5py.File(qcd + "//" + fichiers_h5_qcd[counter_qcd], "r")
                
                    datasetHLF_qcd = un_fichier_qcd['HLF']
                    datasetParticles_qcd = Un_fichier_qcd['Particles']
                
                    hLF.append(list(datasetHLF_qcd[0]))
                    labels.append([1,0,0])
                    particles.append(list(datasetParticles_qcd[0]))
                
                    au_debut_qcd=False
                
                    minicounter_qcd = minicounter_qcd+1
            
                else: #the cursor is in the middle of the files...
                
                    if minicounter_qcd < len(datasetParticles_qcd): #... but not at the end
                
                        hLF.append(list(datasetHLF_qcd[minicounter_qcd]))
                        labels.append([1,0,0])
                        particles.append(list(datasetParticles_qcd[minicounter_qcd]))
                    
                        minicounter_qcd = minicounter_qcd+1
                
                    else: # when the cursor is at the end, the current file is closed and the next one is opened...
                    
                        un_fichier_qcd.close()
                        counter_qcd= counter_qcd+1
                    
                        if counter_qcd < nb_qcd: # ...if there are always files
                    
                            minicounter_qcd=0
                
                            un_fichier_qcd = h5py.File(qcd + "//" + fichiers_h5_qcd[counter_qcd], "r")
                
                            datasetHLF_qcd = un_fichier_qcd['HLF']
                            datasetParticles_qcd = un_fichier_qcd['Particles']
                
                            hLF.append(list(datasetHLF_qcd[0]))
                            labels.append([1,0,0])
                            particles.append(list(datasetParticles_qcd[0]))
                    
                            minicounter_qcd = minicounter_qcd+1
        
            elif choix == 2: #ttbar
            
                if au_debut_ttbar:
                
                    minicounter_ttbar=0
                
                    un_fichier_ttbar = h5py.File(ttbar + "//" + fichiers_h5_ttbar[counter_ttbar], "r")
                
                    datasetHLF_ttbar = un_fichier_ttbar['HLF']
                    datasetParticles_ttbar = un_fichier_ttbar['Particles']
                
                    hLF.append(list(datasetHLF_ttbar[0]))
                    labels.append([0,1,0])
                    particles.append(list(datasetParticles_ttbar[0]))
                
                    au_debut_ttbar=False
                
                    minicounter_ttbar = minicounter_ttbar+1
            
                else:
                
                    if minicounter_ttbar < len(datasetParticles_ttbar):
                
                        hLF.append(list(datasetHLF_ttbar[minicounter_ttbar]))
                        labels.append([0,1,0])
                        particles.append(list(datasetParticles_ttbar[minicounter_ttbar]))
                    
                        minicounter_ttbar = minicounter_ttbar+1
                
                    else:
                    
                        un_fichier_ttbar.close()
                        counter_ttbar= counter_ttbar+1
                    
                        if counter_ttbar < nb_ttbar:
                    
                            minicounter_ttbar=0
                
                            un_fichier_ttbar = h5py.File(ttbar + "//" + fichiers_h5_ttbar[counter_ttbar], "r")
                
                            datasetHLF_ttbar = un_fichier_ttbar['HLF']
                            datasetParticles_ttbar = un_fichier_ttbar['Particles']

                            hLF.append(list(datasetHLF_ttbar[0]))
                            labels.append([0,1,0])
                            particles.append(list(datasetParticles_ttbar[0]))
                    
                            minicounter_ttbar = minicounter_ttbar+1
        
            else: #W
            
                if au_debut_W:
                
                    minicounter_W=0
                
                    un_fichier_W = h5py.File(W + "//" + fichiers_h5_W[counter_W], "r")
                
                    datasetHLF_W = un_fichier_W['HLF']
                    datasetParticles_W = un_fichier_W['Particles']
                
                    hLF.append(list(datasetHLF_W[0]))
                    labels.append([0,0,1])
                    particles.append(list(datasetParticles_W[0]))
                
                    au_debut_W=False
                
                    minicounter_W = minicounter_W+1
            
                else:
                
                    if minicounter_W < len(datasetParticles_W):
                
                        hLF.append(list(datasetHLF_W[minicounter_W]))
                        labels.append([0,0,1])
                        particles.append(list(datasetParticles_W[minicounter_W]))
                    
                        minicounter_W = minicounter_W+1
                
                    else:
                    
                        un_fichier_W.close()
                        counter_W= counter_W+1
                    
                        if counter_W < nb_W:
                    
                            minicounter_W=0
                
                            un_fichier_W = h5py.File(W + "//" + fichiers_h5_W[counter_W], "r")
                
                            datasetHLF_W = un_fichier_W['HLF']
                            datasetParticles_W = un_fichier_W['Particles']
                
                            hLF.append(list(datasetHLF_W[0]))
                            labels.append([0,0,1])
                            particles.append(list(datasetParticles_W[0]))
                    
                            minicounter_W = minicounter_W+1
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
