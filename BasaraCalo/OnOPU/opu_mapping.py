# -*- coding: utf-8 -*-
# This code takes as an argument the files generated by TransformArraySparseFloat.py and outputs their projections
# using the appropriate quartile file. The output directory needs to be changed manually if needed.
# Example usage:
# python opu_mapping.py --type ttbar --ncomp 50000

import argparse
import os

import numpy as np
import pandas as pd

import sparse
from scipy.ndimage.filters import maximum_filter
from tqdm import tqdm

from lightonml.projections.sklearn import OPUMap


def parse_args():
    parser = argparse.ArgumentParser(
        description="HEP OPU - Project image on OPU", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--type", type=str, default="ttbar", help="Process type, ttbar or W"
    )
    parser.add_argument(
        "--ncomp", type=int, default=30000, help="Number of random features produced"
    )
    args = parser.parse_args()

    return args

def opu_projection(type, ncomp):
    print(f'Using {type} filter.')
    quartile_filter = pd.read_csv(f"../quartile_filters_{type}.csv",
                                  index_col=0)

    filenames = ['ECAL_ForwardN', 'ECAL_EBEE', 'ECAL_ForwardP',
                 'ECAL_Gamma_ForwardN', 'ECAL_Gamma_EBEE', 'ECAL_Gamma_ForwardP',
                 'HCAL_ForwardN', 'HCAL_HEN', 'HCAL_HB', 'HCAL_HEP', 'HCAL_ForwardP']

    # Directory containing the outputs of TransformArraySparseFloat.py
    namedir = "../ArrayInputs_900/fmixed_data_"
    print(f'OPU mapping with {ncomp} components.')
    random_mapping = OPUMap(n_components=ncomp,
                            ndims=2)


    def file2arr(filestub):
        '''
        Transform the files of the 'filestub'-th generated process into a
        set of canvas and labels.

        Parameters
        ----------
        filestub: int,
            the number characterizing the process considered. Must be padded
            with zeros to be of length 3

        Returns
        ----------
        canvas: np.ndarray,
            boolean array to be mapped to the DMD
        labels: np.ndarray
        '''
        f = {}

        labels = np.load(namedir + filestub + ".h5Labels.npz")['data']
        nevents = labels.shape[0]
        for i, fn in enumerate(filenames):
            f[fn] = sparse.load_npz(namedir + filestub + ".h5" + fn + ".npz")

        canvas = np.zeros([nevents, 900, 1115], dtype=bool)

        for iimg in range(nevents):
            for ilab, irow in quartile_filter.iterrows():
                img = f[ilab][iimg].todense()
                fimg = maximum_filter(img > irow.qval, irow["filter"])
                if irow.t: fimg = fimg.T
                y = int(irow.y)
                x = int(irow.x)
                if irow.x == 850:
                    y += int(186 * irow.idq)
                else:
                    y += int(372 * irow.idq)
                canvas[iimg, x:x + fimg.shape[0], y:y + fimg.shape[1]] = fimg
        return canvas, labels

    # Output directory, change if needed
    outdir = os.path.join("./opuout/", f"{type}_AllEvtSparse_{ncomp}_randomvariables/")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print("Output directory:", outdir)

    with random_mapping.opu:
        for i, filestubn in enumerate(tqdm(range(260))):
            filestub = f"{filestubn}"
            arr, labels = file2arr(filestub)
            if i == 0:
                OPUoutput = random_mapping.fit_transform(arr)
            else:
                OPUoutput = random_mapping.transform(arr)
            np.savez_compressed(outdir + f"{filestub}.npz",
                                OPU=OPUoutput,
                                labels=labels)


if __name__ == "__main__":
    args = parse_args()
    opu_projection(args.type, args.ncomp)