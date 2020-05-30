import numpy as np
import h5py 
import matplotlib.pyplot as plt 
from utils import *

def main():
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}

if __name__ == "__main__":
    main()