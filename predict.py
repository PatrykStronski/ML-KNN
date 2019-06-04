from libb import classify
import numpy as np

def predict(x):
    return np.array(classify(x,4,0.3))
