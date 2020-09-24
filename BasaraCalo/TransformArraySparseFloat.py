import glob
import h5py
import pandas as pd
import numpy as np
from scipy import misc
import time
import sys
import matplotlib
from skimage import draw 
import os
import matplotlib.pyplot as plt
import random
from array import array
import sparse






features = ['Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi', 
                    'vtxX', 'vtxY', 'vtxZ','ChPFIso', 'GammaPFIso', 'NeuPFIso',
                    'isChHad', 'isNeuHad', 'isGamma', 'isEle',  'isMu', 
                        #'Charge'
           ]




def showSEvent(d,i):
    
    data = d[int(i),...]
    
    all_hists = []

    
    phi_36b=np.linspace(-np.pi, np.pi, 37)
    phi_360b = np.linspace(-np.pi, np.pi, 361)
    phi_72b = np.linspace(-np.pi, np.pi, 73)
    
    lim_eta_endcap=2.958
    eta_bin_width=0.0174
    eta_barrel = np.arange(-85, 86)*eta_bin_width
    eta_endcapN= np.arange(1, 85)*eta_bin_width - lim_eta_endcap
    eta_endcapP= np.arange(1, 85)*eta_bin_width + 1.4964
    eta_ebee = np.concatenate((eta_endcapN, eta_barrel, eta_endcapP))

    eta_forwardN =[-5.001, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -lim_eta_endcap]
    eta_forwardP = [lim_eta_endcap, 3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5.001]

    eta_HB = [-1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.87, -0.783, -0.696, 
            -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0, 0.087, 0.174, 0.261, 
            0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.87, 0.957, 1.044, 1.131, 1.218, 1.305, 
            1.392, 1.479, 1.566]


    ECAL_ForwardN = sparse.DOK((len(eta_forwardN)-1, 36), dtype=np.float)
    ECAL_ForwardP = sparse.DOK((len(eta_forwardP)-1, 36), dtype=np.float)
    ECAL_EBEE =     sparse.DOK((len(eta_ebee)-1, 360), dtype=np.float)

    ECAL_Gamma_ForwardN = sparse.DOK((len(eta_forwardN)-1, 36), dtype=np.float)
    ECAL_Gamma_ForwardP = sparse.DOK((len(eta_forwardP)-1, 36), dtype=np.float)
    ECAL_Gamma_EBEE =     sparse.DOK((len(eta_ebee)-1, 360), dtype=np.float)

    HCAL_ForwardN =  sparse.DOK((len(eta_forwardN)-1, 36), dtype=np.float)
    HCAL_ForwardP =  sparse.DOK((len(eta_forwardP)-1, 36), dtype=np.float)
    HCAL_HEN =       sparse.DOK((len(eta_endcapN)-1, 360), dtype=np.float)
    HCAL_HB =        sparse.DOK((len(eta_HB)-1, 72), dtype=np.float)
    HCAL_HEP =       sparse.DOK((len(eta_endcapP)-1, 360), dtype=np.float)
    
    

    for ip in range(data.shape[0]):
        p_data = data[ip,:]
        eta = p_data[0]
        phi = p_data[1]
#       if eta==0 and phi==0: continue
        pT = p_data[2]
        ptype = int(p_data[3])

        phibin=np.digitize(phi, phi_360b)-1
        if (abs(eta) > lim_eta_endcap): phibin=np.digitize(phi, phi_36b)-1  # Forward 

        if ptype == 1: # NeuHad. Fill HCAL:
            if (abs(eta) < 1.5): #Fill HB
                HCAL_HB[np.digitize(eta, eta_HB)-1,  np.digitize(phi, phi_72b)-1] +=  pT
            elif (abs(eta) < lim_eta_endcap): #Fill HE
                if eta > 0: HCAL_HEP[np.digitize(eta, eta_endcapP)-1,  phibin] +=  pT #Fill HEP
                else:       HCAL_HEN[np.digitize(eta, eta_endcapN)-1,  phibin] +=  pT #Fill HEN
            else: #Fill HF
                phibin=np.digitize(phi, phi_36b)-1
                if eta > 0: HCAL_ForwardP[np.digitize(eta, eta_forwardP)-1,  phibin] +=  pT #Fill forward P
                else:       HCAL_ForwardN[np.digitize(eta, eta_forwardN)-1,  phibin] +=  pT #Fill forward N
        elif ptype == 0 or ptype == 3 or ptype == 4: # Track. Fill ECAL
            if (abs(eta) < lim_eta_endcap):  #Fill ebee
                ECAL_EBEE[np.digitize(eta, eta_ebee)-1, phibin] +=  pT
            else: #Fill ECAL Forward
                if eta > 0: ECAL_ForwardP[np.digitize(eta, eta_forwardP)-1, phibin] +=  pT  #Fill forward P
                else:       ECAL_ForwardN[np.digitize(eta, eta_forwardN)-1,  phibin] +=  pT #Fill forward N
        else: # Gamma
            if (abs(eta) < lim_eta_endcap): ECAL_Gamma_EBEE[np.digitize(eta, eta_ebee)-1,  phibin] +=  pT #Fill ebee
            else: #Fill ECAL Forward
                if eta > 0: ECAL_Gamma_ForwardP[np.digitize(eta, eta_forwardP)-1, phibin] +=  pT #Fill forward P
                else:       ECAL_Gamma_ForwardN[np.digitize(eta, eta_forwardN)-1,  phibin] +=  pT #Fill forward N
                    
                    
    # Convert all hists to numpy arrays and append to all_hists
    all_hists.append(ECAL_ForwardN)
    all_hists.append(ECAL_EBEE)
    all_hists.append(ECAL_ForwardP)
    all_hists.append(ECAL_Gamma_ForwardN)
    all_hists.append(ECAL_Gamma_EBEE)
    all_hists.append(ECAL_Gamma_ForwardP)
    all_hists.append(HCAL_ForwardN)
    all_hists.append(HCAL_HEN)
    all_hists.append(HCAL_HB)
    all_hists.append(HCAL_HEP)
    all_hists.append(HCAL_ForwardP)
    
    return all_hists

def do_it_all( sample ,limit=1e6 ):
    start = time.mktime(time.gmtime())
    dataset = {}
    N=1000
    Nsample=sample.shape[0]
    max_I = np.min([limit, Nsample])
    print("Max samples :", max_I)
    for i in range(max_I):
        if i%N==0: 
            now = time.mktime(time.gmtime())
            so_far = now-start
            if i:
                eta = (so_far/i* max_I) - so_far
                print (i, so_far,"[s] ; finishing in", int(eta),"[s]", int(eta/60.),"[m]")
        all_hists = showSEvent(sample, i)
        
        # Return 11 numpy arrays corresponding to 11 histograms

        
        for hist in range(len(all_hists)):
            if i==0 : dataset[hist]=[]
            dataset[hist].append ( all_hists[hist].to_coo() )
    return dataset



def make_reduced( f ) :
    if type(f) == str:
        f = h5py.File(f)    
    pf = f['Particles']
    reduced = np.zeros( (pf.shape[0], 801, 4))
    reduced[...,0] = f['Particles'][...,features.index('Eta')] 
    reduced[...,1] = f['Particles'][...,features.index('Phi')] 
    reduced[...,2] = np.minimum(np.log(np.maximum(f['Particles'][...,features.index('Pt')], 1.001))/5., 10)
    reduced[...,3] = np.argmax( f['Particles'][..., 13:], axis=-1)

    h_reduced = np.zeros( (pf.shape[0], 1, 4))
    h_reduced[...,0,2] = np.minimum(np.maximum(np.log(f['HLF'][..., 1])/5.,0.001), 10) # MET
    h_reduced[...,0,1] = f['HLF'][..., 2] # MET-phi
    h_reduced[...,0,3 ] = int(5) ## met type

    reduced = np.concatenate( (reduced, h_reduced), axis=1)

    return reduced

def convert_sample( inFileName, limit=None ):
    f = h5py.File(inFileName, "r")    
    reduced = make_reduced(f)

    outFileName = "AllArrayOutput/f"+inFileName.rsplit('/', 1)[-1]
    if limit:
        print ("Converting",inFileName,"into",outFileName,("for %s events"%limit)) 
    ds = do_it_all( reduced ,limit)

   
    filenames=['ECAL_ForwardN', 'ECAL_EBEE', 'ECAL_ForwardP',
        'ECAL_Gamma_ForwardN', 'ECAL_Gamma_EBEE', 'ECAL_Gamma_ForwardP', 
        'HCAL_ForwardN', 'HCAL_HEN', 'HCAL_HB', 'HCAL_HEP', 'HCAL_ForwardP']

    for i in range(len(filenames)):
        fn=outFileName+filenames[i]+".npz"
        data=sparse.stack(ds[i])
        print(fn, data.shape)
        sparse.save_npz(fn, data)

    tmp = f['Labels'][:limit,...] if limit else f['Labels'][...]
    np.savez(outFileName+"Labels.npz", data = tmp, dtype = np.bool)




    
if __name__ == "__main__":
    if len(sys.argv)>1:
        ## make a special file
        limit = int(sys.argv[2]) if len(sys.argv)>2 else None
        convert_sample(sys.argv[1], limit)
    else:
        fl = glob.glob('../ParticleList/*.h5')
        #fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/train/*.h5'))
        #fl.extend(glob.glob('/bigdata/shared/Delphes/np_datasets_new/3_way/MaxLepDeltaR_des/val/*.h5'))
        random.shuffle( fl )
        every = 5
        N= None
        for i,fn in enumerate(fl):
            com = 'python3 TransformArray.py %s'%( fn)
            if N: com += ' %d'%N
            wait = (i%every==(every-1))
            if not wait: com +='&'
            print (com)
            os.system(com)
            if wait and N:
                time.sleep( 60 )

