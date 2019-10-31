import numpy as np


"""
split_data: function that partitions a data set into a training and test set
    Inputs: x,y:  original data set
            test_percent: percentage of data to be used for testing
                          (default is 30%)

    Outputs: x_train,y_train,x_test,y_test: partitioned data
"""
def split_data(x,y,test_percent = .3):
    n = len(y)
    split = int(np.round(test_percent*n))
    
    x_train = x[:-split]
    x_test = x[-split,:]
    y_train = y[:-split]
    y_test = y[-split,:]

    return x_train,y_train,x_test,y_test
