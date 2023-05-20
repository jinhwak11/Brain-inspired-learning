import sys
import struct
import numpy as np
import sklearn
from sklearn import preprocessing

def readChoirData(dir,data_path, data_type):

    filename = dir + data_path + "_" + data_type + ".choir_dat"
    param = dict()

    with open(filename, 'rb') as f:
        nFeatures = struct.unpack('i', f.read(4))[0]
        nClasses = struct.unpack('i', f.read(4))[0]
        X = []
        y = []
        while True:
            newDP = []
            for i in range(nFeatures):
                v_in_bytes = f.read(4)
                if v_in_bytes is None or len(v_in_bytes) == 0:
                    # TODO very unprofessionally normalizing data
                    X = preprocessing.normalize(np.asarray(X), norm='l2')
                    param["nFeatures"], param["nClasses"], param["data"], param["labels"] = \
                        nFeatures, nClasses, X, np.asarray(y)
                    return param
                v = struct.unpack('f', v_in_bytes)[0]
                newDP.append(v)
            l = struct.unpack('i', f.read(4))[0]
            X.append(newDP)
            y.append(l)
    return X,y