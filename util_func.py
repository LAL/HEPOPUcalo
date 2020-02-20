import numpy as np
from math import sqrt,log
import matplotlib.pyplot as plt

def compare_train_test(y_pred_train, y_train, y_pred, y_test, high_low=(0,1), bins=30):
    plt.hist(y_pred_train[y_train == 1],
                 color='r', alpha=0.5, range=high_low, bins=bins,
                 histtype='stepfilled', normed=True,
                 label='S (train)') # alpha is transparancy
    plt.hist(y_pred_train[y_train == 0],
                 color='b', alpha=0.5, range=high_low, bins=bins,
                 histtype='stepfilled', normed=True,
                 label='B (train)')

    hist, bins = np.histogram(y_pred[y_test == 1],
                                  bins=bins, range=high_low, normed=True)
    scale = len(y_pred[y_test == 1]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    #width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(y_pred[y_test == 0],
                                  bins=bins, range=high_low, normed=True)
    scale = len(y_pred[y_test == 0]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    #width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')
    plt.xlabel("NN scores")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')

def shift_max_energy_to_front(images):
    im = np.zeros(images[0].shape)
    for i in range(images.shape[0]):
    
    	sum_arr = np.sum(images[i],axis=0)
    	max_loc = np.argmax(sum_arr)
    	#print(max_loc)
    	new_arrangement = np.arange(0,64)
    	new_arrangement = ((new_arrangement +max_loc)%64).astype(int)
    	for j in range(64):
    	    im[j] = images[i,j,new_arrangement]
    	images[i]=im
    return images
