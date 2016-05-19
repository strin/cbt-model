import os
import numpy as np
import numpy.linalg as npla
import numpy.random as npr
import theano
import theano.tensor as T

if theano.config.floatX == 'float32':
    FX = np.float32
else:
    FX = np.float64
floatX = FX
