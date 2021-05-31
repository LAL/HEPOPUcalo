# -*- coding: utf-8 -*-
# This code takes as an argument the files generated by Mapping_OPUwithEvtID.py and outputs a train/test split with the desired
# amount of events and random projections.
# Example of usage:
# python Splitter.py --type W --inputdir ./opuout/W_AllEvtSparse_100000_randomvariables/
# --ncomp 30000 --nevents 20000
# The output are saved in an automatically created repository. The location of this new repository must be changed
# manually if needed.

import time
from os.path import exists
import os
import argparse
import datetime

import numpy as np
import pandas as pd


from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeCV, SGDClassifier
from sklearn.utils.class_weight import compute_sample_weight
from joblib import dump

from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="HEP OPU - Model Fitting", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--type", type=str, default="ttbar", help="Process type, ttbar or W"
    )
    parser.add_argument(
        "--inputdir", type=str, required=True, help="Path to the output of the OPU files to be used"
    )
    parser.add_argument(
        "--ncomp", type=int, default=30000, help="Number of random features considered"
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=60000,
        help="Number of training samples",
    )
    args = parser.parse_args()

    return args

def import_data(process_type, inputdir, ncomp, nevents):
    eQCD, ett, eW = range(3)
    nfiles = 100
    assert (
        nfiles * 900 > nevents
    ), "Not reading enough files, you must manually change nfiles"
    samples = range(nfiles)
   
    opu_ls = []
    labels_ls = []
    eventID_ls=[]#

    for i in tqdm(samples):
        outpath = inputdir + f"{i}.npz"
        outfile = np.load(outpath)
        opu_ls.append(outfile["OPU"])
        labels_ls.append(outfile["labels"])
        eventID_ls.append(outfile["eventID"])#

    opu_out = np.vstack(opu_ls).astype(np.int16)
    labels_out = np.vstack(labels_ls)
    eventID_out = np.vstack(eventID_out)#
    
    print(f"OPU outputs shape {opu_out.shape}.\n" f"Labels shape {labels_out.shape}")
    assert (
        opu_out.shape[1] > 2 * ncomp
    ), "Not enough random features in the opu outputs considered"

    X =  opu_out[:nevents, :ncomp] - opu_out[:nevents, ncomp: 2 * ncomp]
    y = np.argmax(labels_out[:nevents], axis=1)
    evid = eventID_out[:nevents]#

    test_prop = np.array([0.362, 0.003, 0.635])
    
    if process_type == "ttbar":
        ybool = y == ett
        weight_1 = 2 * test_prop[ett]
    elif process_type == "W":
        ybool = y == eW
        weight_1 = 2 * test_prop[eW]
    weight_0 = 2 - weight_1
    
    ybin=[]#
    
    for i in range(len(ybool)):#
        if ybool[i]:
            ybin.append([1,evid[i]])
        else:
            ybin.append([0,evid[i]])
        
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, ybin, test_size=0.1, stratify=ybool
    )
    
    y_temp_train = y_train[:][0]#
    y_temp_test =y_test[:][0]#
    
    evid_train=y_train[:][1]#
    evid_test=y_test[:][1]#
    
    y_train=[]#
    y_test=[]#
    
    for i in range(len(y_temp_train)):#
        if y_temp_train==0:
            y_train.append(False)
        else:
            y_train.append(True)
            
    for i in range(len(y_temp_test)):#
        if y_temp_test==0:
            y_test.append(False)
        else:
            y_test.append(True)
            
    evid_train=np.array(evid_train)#
    evid_test=np.array(evid_test)
    
    y_train=np.array(y_train)#
    y_test=np.array(y_test)
        
    w1 = compute_sample_weight(class_weight="balanced", y=y_test)
    w2 = compute_sample_weight(class_weight={0:weight_0, 1:weight_1}, y=y_test)
    
    return X_train, y_train, X_test, y_test, w1*w2, evid_train, evid_test


def save_split(process_type, X_train, y_train, X_test, y_test, sample_weight,evid_train, evid_test):#
    # Creating a new repository for the experiments. Change the path to this repository if needed.
    current_time = datetime.datetime.now()
    timestr = current_time.strftime("%d%m%Y_%H%M%S")

    nevents = X_train.shape[0] + X_test.shape[0]
    ncomp = X_train.shape[1]

    exp_path = f"../results/{process_type}_{nevents}evts_{ncomp}f_{timestr}/"
    os.makedirs(exp_path)
    np.savez(
        exp_path + "traintest.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        weights=sample_weight,
        evid_train=evid_train,#
        evid_test=evid_test#
    )

def run_script(process_type, inputdir, ncomp, nevents):
    print("Importing data...")
    X_train, y_train, X_test, y_test, weights,evid_train, evid_test  = import_data(process_type, inputdir, ncomp, nevents)
    print("Splitting...")
    save_split(process_type, X_train, y_train, X_test, y_test, weights, evid_train, evid_test)

if __name__ == "__main__":
    args = parse_args()
    run_script(args.type, args.inputdir, args.ncomp, args.nevents)